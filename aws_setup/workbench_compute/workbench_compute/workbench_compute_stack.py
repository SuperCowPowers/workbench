from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecr as ecr,
    aws_batch as batch,
    aws_sqs as sqs,
    aws_lambda as lambda_,
    aws_lambda_event_sources as lambda_events,
    aws_logs as logs,
    Duration,
    Size,
)
from constructs import Construct
from typing import Any, List, Dict
from dataclasses import dataclass, field
from textwrap import dedent


@dataclass
class WorkbenchComputeStackProps:
    workbench_bucket: str
    batch_role_arn: str
    lambda_role_arn: str
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
        self.ml_pipeline_queue = self.create_ml_pipeline_queue()
        self.batch_trigger_lambda = self.create_batch_trigger_lambda()

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
            tag="0.1",
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
                timeout=Duration.hours(3),
            )
        return job_definitions

    ###########################
    #  ML Pipeline SQS Queue  #
    ###########################
    def create_ml_pipeline_queue(self) -> sqs.Queue:
        """Create SQS FIFO queue for ML pipeline orchestration with deduplication."""

        # Dead letter queue for failed messages (also FIFO)
        dlq = sqs.Queue(
            self,
            "MLPipelineDLQ",
            queue_name="workbench-ml-pipeline-dlq.fifo",  # Must end with .fifo
            fifo=True,
            retention_period=Duration.days(14),
        )

        # Main FIFO queue with deduplication
        return sqs.Queue(
            self,
            "MLPipelineQueue",
            queue_name="workbench-ml-pipeline-queue.fifo",  # Must end with .fifo
            fifo=True,
            content_based_deduplication=True,  # Auto-dedupe based on message content
            visibility_timeout=Duration.minutes(15),
            retention_period=Duration.days(1),
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=1, queue=dlq),
        )

    def create_batch_trigger_lambda(self) -> lambda_.Function:
        """Create Lambda function to process SQS messages and trigger Batch jobs."""

        # Create a mapping of job definition names to pass to Lambda
        job_def_names = {size: f"workbench-batch-{size}" for size in ["small", "medium", "large"]}

        batch_trigger_lambda = lambda_.Function(
            self,
            "BatchTriggerLambda",
            function_name="workbench-batch-trigger",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.lambda_handler",
            code=lambda_.Code.from_inline(self._get_lambda_code()),
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
                self.ml_pipeline_queue, batch_size=1, max_concurrency=5  # Limit parallel Batch job submissions
            )
        )
        return batch_trigger_lambda

    @staticmethod
    def _get_lambda_code() -> str:
        """Return the Lambda function code as a string."""
        return dedent(
            '''
            import json
            import boto3
            import os
            from datetime import datetime
            from pathlib import Path

            batch = boto3.client('batch')
            s3 = boto3.client('s3')

            WORKBENCH_BUCKET = os.environ['WORKBENCH_BUCKET']
            JOB_QUEUE = os.environ['JOB_QUEUE']
            JOB_DEFINITIONS = {
                'small': os.environ['JOB_DEF_SMALL'],
                'medium': os.environ['JOB_DEF_MEDIUM'],
                'large': os.environ['JOB_DEF_LARGE'],
            }

            def lambda_handler(event, context):
                """Process SQS messages and submit Batch jobs."""

                for record in event['Records']:
                    try:
                        message = json.loads(record['body'])
                        script_path = message['script_path']  # s3://bucket/path/to/script.py
                        size = message.get('size', 'small')
                        extra_env = message.get('environment', {})
                        model_outputs = message.get('MODEL_OUTPUTS', [])
                        endpoint_outputs = message.get('ENDPOINT_OUTPUTS', [])

                        script_name = Path(script_path).stem
                        job_name = f"workbench_{script_name}_{datetime.now():%Y%m%d_%H%M%S}"

                        # Get job definition name from environment variables
                        job_def_name = JOB_DEFINITIONS.get(size, JOB_DEFINITIONS['small'])

                        # Build environment variables
                        env_vars = [
                            {'name': 'ML_PIPELINE_S3_PATH', 'value': script_path},
                            {'name': 'WORKBENCH_BUCKET', 'value': WORKBENCH_BUCKET},
                            *[{'name': k, 'value': v} for k, v in extra_env.items()]
                        ]

                        # Add MODEL_OUTPUTS as comma-separated list if provided
                        if model_outputs:
                            env_vars.append({'name': 'MODEL_OUTPUTS', 'value': ','.join(model_outputs)})

                        # Add ENDPOINT_OUTPUTS as comma-separated list if provided
                        if endpoint_outputs:
                            env_vars.append({'name': 'ENDPOINT_OUTPUTS', 'value': ','.join(endpoint_outputs)})

                        response = batch.submit_job(
                            jobName=job_name,
                            jobQueue=JOB_QUEUE,
                            jobDefinition=job_def_name,
                            containerOverrides={
                                'environment': env_vars
                            }
                        )

                        print(f"Submitted job: {job_name} ({response['jobId']})")

                    except Exception as e:
                        print(f"Error processing message: {e}")
                        raise  # Let SQS retry via DLQ

                return {'statusCode': 200}
        '''
        ).strip()
