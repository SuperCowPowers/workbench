from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecr as ecr,
    aws_batch as batch,
    aws_sqs as sqs,
    aws_sns as sns,
    aws_lambda as lambda_,
    aws_lambda_event_sources as lambda_events,
    aws_events as events,
    aws_events_targets as targets,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    Duration,
    Size,
)
from constructs import Construct
from typing import Any, List, Dict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkbenchComputeStackProps:
    workbench_bucket: str
    batch_role_arn: str
    lambda_role_arn: str
    environment_name: str = "unknown"  # Environment name (sandbox, dev, stage, prod)
    existing_vpc_id: str = None
    subnet_ids: List[str] = field(default_factory=list)  # Optional subnets ids for the Batch compute environment


class WorkbenchComputeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        props: WorkbenchComputeStackProps,
        **kwargs: Any,
    ) -> None:
        desc = "Workbench Compute: Defines the compute infrastructure for Workbench including Batch resources."
        super().__init__(scope, construct_id, env=env, description=desc, **kwargs)

        # Print the environment details
        print("Environment")
        print(env)

        # Grab our properties
        self.workbench_bucket = props.workbench_bucket
        self.batch_role_arn = props.batch_role_arn
        self.environment_name = props.environment_name
        self.existing_vpc_id = props.existing_vpc_id
        self.subnet_ids = props.subnet_ids

        # Import the Batch role
        self.workbench_batch_role = iam.Role.from_role_arn(self, "ImportedBatchRole", self.batch_role_arn)

        # Import the Lambda role
        self.workbench_lambda_role = iam.Role.from_role_arn(self, "ImportedLambdaRole", props.lambda_role_arn)

        # Create Batch resources first (Lambda will reference them)
        self.batch_compute_environment = self.create_batch_compute_environment()
        self.batch_job_queue = self.create_batch_job_queue()
        self.batch_job_definitions = self.create_batch_job_definitions()

        # Create ML Pipeline SQS Queue and Lambda (after Batch resources)
        self.ml_pipeline_dlq, self.ml_pipeline_queue = self.create_ml_pipeline_queue()
        self.batch_trigger_lambda = self.create_batch_trigger_lambda()

        # Create Batch failure notification resources
        self.batch_failure_topic = self.create_batch_failure_topic()
        self.batch_failure_lambda = self.create_batch_failure_lambda()
        self.create_batch_failure_rule()

        # Create DLQ alarm (uses the batch failure topic for notifications)
        self.create_dlq_alarm()

    #####################
    #   Batch Compute   #
    #####################
    def create_batch_compute_environment(self) -> batch.FargateComputeEnvironment:
        """Create the Batch compute environment with Fargate."""

        # Check if we have an existing VPC to use
        if self.existing_vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "ImportedVPC", vpc_id=self.existing_vpc_id)
            vpc_subnets = None

            # Use specific subnets if provided
            if self.subnet_ids:
                vpc_subnets = ec2.SubnetSelection(
                    subnets=[
                        ec2.Subnet.from_subnet_id(self, f"BatchSubnet{i}", subnet_id)
                        for i, subnet_id in enumerate(self.subnet_ids)
                    ]
                )
        else:
            raise ValueError(
                'Please provide the Workbench Config entry: "WORKBENCH_VPC_ID":"vpc-123456789abcdef0". '
                "Default VPC networks cannot pull ECR images for Batch jobs."
            )

        return batch.FargateComputeEnvironment(
            self,
            "WorkbenchBatchComputeEnvironment",
            compute_environment_name="workbench-compute-env",
            vpc=vpc,
            vpc_subnets=vpc_subnets,
            maxv_cpus=16,  # Limit to 16 vCPU to minimize AWS Throttling issues
        )

    def create_batch_job_queue(self) -> batch.JobQueue:
        """Create the Batch job queue."""
        return batch.JobQueue(
            self,
            "WorkbenchBatchJobQueue",
            job_queue_name="workbench-job-queue",
            compute_environments=[
                batch.OrderedComputeEnvironment(compute_environment=self.batch_compute_environment, order=1)
            ],
        )

    def create_batch_job_definitions(self) -> Dict[str, batch.EcsJobDefinition]:
        """Create ML pipeline job definitions in small/medium/large tiers."""

        # Using ECR image for ML pipelines
        ecr_image = ecs.ContainerImage.from_ecr_repository(
            repository=ecr.Repository.from_repository_arn(
                self,
                "MLPipelineRepo",
                repository_arn=f"arn:aws:ecr:{self.region}:507740646243:repository/aws-ml-images/py312-ml-pipelines",
            ),
            tag="0.3",
        )

        # Job Definition Tiers
        tiers = {
            "small": (2, 4096),  # 2 vCPU, 4GB RAM
            "medium": (4, 8192),  # 4 vCPU, 8GB RAM
            "large": (8, 16384),  # 8 vCPU, 16GB RAM
        }

        job_definitions = {}
        for size, (cpu, memory_mib) in tiers.items():
            job_definitions[size] = batch.EcsJobDefinition(
                self,
                f"JobDef{size.capitalize()}",
                job_definition_name=f"workbench-batch-{size}",
                container=batch.EcsFargateContainerDefinition(
                    self,
                    f"Container{size.capitalize()}",
                    cpu=cpu,
                    memory=Size.mebibytes(memory_mib),
                    image=ecr_image,
                    job_role=self.workbench_batch_role,
                    execution_role=self.workbench_batch_role,
                    environment={
                        "WORKBENCH_BUCKET": self.workbench_bucket,
                        "PYTHONUNBUFFERED": "1",
                    },
                ),
                timeout=Duration.hours(6),
            )
        return job_definitions

    ###########################
    #  ML Pipeline SQS Queue  #
    ###########################
    def create_ml_pipeline_queue(self) -> tuple[sqs.Queue, sqs.Queue]:
        """Create SQS FIFO queue for ML pipeline orchestration with deduplication.

        Returns:
            Tuple of (dlq, main_queue)
        """

        # Dead letter queue for failed messages (also FIFO)
        dlq = sqs.Queue(
            self,
            "MLPipelineDLQ",
            queue_name="workbench-ml-pipeline-dlq.fifo",  # Must end with .fifo
            fifo=True,
            retention_period=Duration.days(14),
        )

        # Main FIFO queue with deduplication
        main_queue = sqs.Queue(
            self,
            "MLPipelineQueue",
            queue_name="workbench-ml-pipeline-queue.fifo",  # Must end with .fifo
            fifo=True,
            content_based_deduplication=True,  # Auto-dedupe based on message content
            visibility_timeout=Duration.minutes(15),
            retention_period=Duration.days(1),
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=1, queue=dlq),
        )

        return dlq, main_queue

    def create_batch_trigger_lambda(self) -> lambda_.Function:
        """Create Lambda function to process SQS messages and trigger Batch jobs.

        The Lambda handles job dependencies by:
        1. Parsing WORKBENCH_BATCH config from scripts in S3
        2. Querying Batch for active jobs in the same group
        3. Submitting with dependsOn to ensure proper execution order
        """

        # Create a mapping of job definition names to pass to Lambda
        job_def_names = {size: f"workbench-batch-{size}" for size in ["small", "medium", "large"]}

        # Path to the Lambda code directory
        lambda_code_path = Path(__file__).parent / "lambdas" / "batch_trigger"

        batch_trigger_lambda = lambda_.Function(
            self,
            "BatchTriggerLambda",
            function_name="workbench-batch-trigger",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(lambda_code_path)),
            timeout=Duration.minutes(5),
            log_retention=logs.RetentionDays.ONE_WEEK,
            environment={
                "WORKBENCH_BUCKET": self.workbench_bucket,
                "JOB_QUEUE": self.batch_job_queue.job_queue_name,
                "JOB_DEF_SMALL": job_def_names["small"],
                "JOB_DEF_MEDIUM": job_def_names["medium"],
                "JOB_DEF_LARGE": job_def_names["large"],
            },
            role=self.workbench_lambda_role,
        )

        # Connect SQS to Lambda
        batch_trigger_lambda.add_event_source(
            lambda_events.SqsEventSource(
                self.ml_pipeline_queue, batch_size=1  # One message at a time for dependency tracking
            )
        )
        return batch_trigger_lambda

    ################################
    #  Batch Failure Notification  #
    ################################
    def create_batch_failure_topic(self) -> sns.Topic:
        """Create SNS topic for batch job failure notifications."""
        topic = sns.Topic(
            self,
            "BatchJobFailureTopic",
            topic_name="workbench-batch-job-failure",
            display_name="Workbench Batch Job Failure",
        )

        # Allow CloudWatch Alarms to publish to this topic
        topic.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudWatchAlarms",
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("cloudwatch.amazonaws.com")],
                actions=["sns:Publish"],
                resources=[topic.topic_arn],
                conditions={"ArnLike": {"aws:SourceArn": f"arn:aws:cloudwatch:{self.region}:{self.account}:alarm:*"}},
            )
        )

        return topic

    def create_batch_failure_lambda(self) -> lambda_.Function:
        """Create Lambda function to handle batch job failures and send notifications."""

        # Path to the Lambda code directory
        lambda_code_path = Path(__file__).parent / "lambdas" / "batch_failure"

        batch_failure_lambda = lambda_.Function(
            self,
            "BatchFailureLambda",
            function_name="workbench-batch-failure",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(lambda_code_path)),
            timeout=Duration.minutes(1),
            log_retention=logs.RetentionDays.ONE_WEEK,
            environment={
                "WORKBENCH_ENVIRONMENT": self.environment_name,
                "BATCH_FAILURE_TOPIC_ARN": self.batch_failure_topic.topic_arn,
            },
            role=self.workbench_lambda_role,
        )

        # Grant permission to publish to SNS
        self.batch_failure_topic.grant_publish(batch_failure_lambda)

        # Grant permission to read CloudWatch logs
        batch_failure_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["logs:GetLogEvents"],
                resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/batch/job:*"],
            )
        )

        return batch_failure_lambda

    def create_batch_failure_rule(self) -> events.Rule:
        """Create EventBridge rule to trigger batch failure lambda on job failures."""

        batch_failure_rule = events.Rule(
            self,
            "BatchJobFailureRule",
            rule_name="workbench-batch-job-failure",
            event_pattern=events.EventPattern(
                source=["aws.batch"],
                detail_type=["Batch Job State Change"],
                detail={
                    "status": ["FAILED"],
                    "jobQueue": [f"arn:aws:batch:{self.region}:{self.account}:job-queue/workbench-job-queue"],
                },
            ),
            description="Triggers batch failure notification when Workbench Batch jobs fail",
        )

        batch_failure_rule.add_target(targets.LambdaFunction(self.batch_failure_lambda))

        return batch_failure_rule

    def create_dlq_alarm(self) -> cloudwatch.Alarm:
        """Create CloudWatch alarm for DLQ messages to alert on pipeline failures."""

        alarm = cloudwatch.Alarm(
            self,
            "MLPipelineDLQAlarm",
            alarm_name="workbench-ml-pipeline-dlq-alarm",
            alarm_description="Alert when messages appear in the ML Pipeline dead letter queue",
            metric=self.ml_pipeline_dlq.metric_approximate_number_of_messages_visible(
                period=Duration.minutes(15),
                statistic="Maximum",
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # Send alarm notifications to the batch failure topic
        alarm.add_alarm_action(cw_actions.SnsAction(self.batch_failure_topic))

        return alarm
