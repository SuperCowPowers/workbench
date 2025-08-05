from aws_cdk import Environment, Stack, aws_iam as iam, aws_logs as logs, RemovalPolicy
from constructs import Construct
from typing import Any, List
from dataclasses import dataclass, field


@dataclass
class WorkbenchCoreStackProps:
    workbench_bucket: str
    workbench_role_name: str
    sso_group: str
    additional_buckets: List[str] = field(default_factory=list)


class WorkbenchCoreStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        props: WorkbenchCoreStackProps,
        **kwargs: Any,
    ) -> None:
        desc = "Workbench Core: Workbench-ExecutionRole(API),  Workbench-GlueRole, and Workbench-LambdaRole"
        super().__init__(scope, construct_id, description=desc, **kwargs)

        # Grab our properties
        self.workbench_bucket = props.workbench_bucket
        self.workbench_role_name = props.workbench_role_name
        self.sso_group = props.sso_group
        self.additional_buckets = props.additional_buckets

        # Create the WorkbenchLogGroup for CloudWatch Logs
        self.workbench_log_group = logs.LogGroup(
            self,
            "WorkbenchLogGroup",
            log_group_name="WorkbenchLogGroup",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Create a list of buckets
        athena_bucket = "aws-athena-query-results*"
        sagemaker_bucket = "sagemaker-{region}-{account_id}*"
        self.bucket_list = [self.workbench_bucket, athena_bucket, sagemaker_bucket] + self.additional_buckets
        self.bucket_arns = self._bucket_names_to_arns(self.bucket_list)

        # Create our managed polices
        self.datasource_policy = self.workbench_datasource_policy()
        self.featureset_policy = self.workbench_featureset_policy()
        self.model_policy = self.workbench_model_policy()
        self.endpoint_policy = self.workbench_endpoint_policy()
        self.pipeline_policy = self.workbench_pipeline_policy()

        # Create our main Workbench Execution Role
        self.workbench_api_execution_role = self.create_api_execution_role()

        # Create additional roles for Lambda and Glue
        self.workbench_lambda_role = self.create_lambda_role()
        self.workbench_glue_role = self.create_glue_role()

    ####################
    #    S3 Buckets    #
    ####################
    def _bucket_names_to_arns(self, bucket_list: list[str]) -> list[str]:
        """Convert a list of dynamic bucket names to ARNs."""
        arns = []
        for bucket_name_template in bucket_list:
            # Dynamically construct the bucket name
            bucket_name = bucket_name_template.format(region=self.region, account_id=self.account)
            bucket_arn = f"arn:aws:s3:::{bucket_name}"
            arns.append(bucket_arn)
            arns.append(f"{bucket_arn}/*")
        return arns

    def s3_read(self) -> iam.PolicyStatement:
        """Create a policy statement for S3 read access."""
        return iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetBucketAcl",
            ],
            resources=self.bucket_arns,
        )

    def s3_full(self) -> iam.PolicyStatement:
        """Create a policy statement for S3 full access."""
        read_statement = self.s3_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "s3:PutObject",
                "s3:DeleteObject",
            ],
            resources=self.bucket_arns,
        )

    @staticmethod
    def s3_public() -> iam.PolicyStatement:
        """Create a policy statement for access to PUBLIC S3 buckets.

        Returns:
            iam.PolicyStatement: The policy statement for access to PUBLIC S3 buckets.
        """
        return iam.PolicyStatement(
            actions=[
                # Define the S3 actions you need
                "s3:GetObject",
                "s3:ListBucket",
            ],
            resources=["arn:aws:s3:::*"],
        )

    ######################
    #    Glue Catalog    #
    ######################
    def glue_catalog_read(self) -> iam.PolicyStatement:
        """Read-only discovery across the entire Glue Data Catalog."""
        return iam.PolicyStatement(
            actions=[
                "glue:GetDatabase",
                "glue:GetDatabases",
                "glue:GetTable",
                "glue:GetTables",
                "glue:SearchTables",
            ],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog"],
        )

    def glue_catalog_full(self) -> iam.PolicyStatement:
        """Full catalog access including database creation."""
        read_statement = self.glue_catalog_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "glue:CreateDatabase",
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
            ],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog"],
        )

    #######################
    #    Glue Database    #
    #######################
    def glue_databases_read(self) -> iam.PolicyStatement:
        """Read-only access to Workbench-managed databases and tables."""
        return iam.PolicyStatement(
            actions=[
                "glue:GetDatabase",
                "glue:GetTable",
                "glue:GetTables",
                "glue:GetPartition",
                "glue:GetPartitions",
            ],
            resources=self._workbench_database_arns(),
        )

    def glue_databases_full(self) -> iam.PolicyStatement:
        """Full access to Workbench-managed databases and tables."""
        read_statement = self.glue_databases_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
            ],
            resources=self._workbench_database_arns(),
        )

    def _workbench_database_arns(self) -> list[str]:
        """Helper to get all Workbench-managed database/table ARNs."""
        return [
            f"arn:aws:glue:{self.region}:{self.account}:database/workbench",
            f"arn:aws:glue:{self.region}:{self.account}:table/workbench/*",
            f"arn:aws:glue:{self.region}:{self.account}:database/sagemaker_featurestore",
            f"arn:aws:glue:{self.region}:{self.account}:table/sagemaker_featurestore/*",
            f"arn:aws:glue:{self.region}:{self.account}:database/inference_store",
            f"arn:aws:glue:{self.region}:{self.account}:table/inference_store/*",
        ]

    #####################
    #     Glue Jobs     #
    #####################
    def glue_pass_role(self) -> iam.PolicyStatement:
        """Allows us to specify the Workbench-Glue role when creating a Glue Job"""
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[f"arn:aws:iam::{self.account}:role/Workbench-GlueRole"],
            conditions={"StringEquals": {"iam:PassedToService": "glue.amazonaws.com"}},
        )

    @staticmethod
    def glue_jobs_discover() -> iam.PolicyStatement:
        """Discovery access to list all Glue jobs."""
        return iam.PolicyStatement(
            actions=["glue:GetJobs"],
            resources=["*"],
        )

    def glue_jobs_read(self) -> iam.PolicyStatement:
        """Read-only access to specific Glue jobs."""
        return iam.PolicyStatement(
            actions=[
                "glue:GetJob",
                "glue:GetJobRun",
                "glue:GetJobRuns",
            ],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:job/*"],
        )

    def glue_jobs_full(self) -> iam.PolicyStatement:
        """Full access to specific Glue jobs."""
        read_statement = self.glue_jobs_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "glue:CreateJob",
                "glue:UpdateJob",
                "glue:StartJobRun",
            ],
            resources=read_statement.resources,
        )

    ####################
    #      Athena      #
    ####################
    @staticmethod
    def athena_read() -> iam.PolicyStatement:
        """Read-only access to Athena queries and workgroups."""
        return iam.PolicyStatement(
            actions=[
                # Query actions
                "athena:ListQueryExecutions",
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:StopQueryExecution",
                # Workgroup actions
                "athena:GetWorkGroup",
                "athena:ListWorkGroups",
            ],
            resources=["*"],
        )

    ######################
    #    FeatureStore    #
    ######################
    @staticmethod
    def featurestore_discovery() -> iam.PolicyStatement:
        """Discovery - list all SageMaker feature groups."""
        return iam.PolicyStatement(
            actions=["sagemaker:ListFeatureGroups"],
            resources=["*"],  # Required for listing operations
        )

    def featurestore_read(self) -> iam.PolicyStatement:
        """Read-only access to SageMaker feature groups."""
        return iam.PolicyStatement(
            actions=[
                "sagemaker:DescribeFeatureGroup",
                "sagemaker:GetRecord",
                "sagemaker:ListTags",
            ],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:feature-group/*"],
        )

    def featurestore_full(self) -> iam.PolicyStatement:
        """Full CRUD access to SageMaker feature groups."""
        read_statement = self.featurestore_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "sagemaker:CreateFeatureGroup",
                "sagemaker:DeleteFeatureGroup",
                "sagemaker:PutRecord",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=read_statement.resources,
        )

    ######################
    #       Models       #
    ######################
    @staticmethod
    def models_discovery() -> iam.PolicyStatement:
        """Discovery - list all SageMaker models, model packages, and model package groups."""
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListModels",
                "sagemaker:ListModelPackages",
                "sagemaker:ListModelPackageGroups",
            ],
            resources=["*"],  # Required for listing operations
        )

    def models_read(self) -> iam.PolicyStatement:
        """Read-only access to SageMaker models and model packages."""
        return iam.PolicyStatement(
            actions=[
                "sagemaker:DescribeModel",
                "sagemaker:DescribeModelPackage",
                "sagemaker:DescribeModelPackageGroup",
                "sagemaker:GetModelPackage",
                "sagemaker:GetModelPackageGroup",
                "sagemaker:ListTags",
            ],
            resources=[
                f"arn:aws:sagemaker:{self.region}:{self.account}:model/*",
                f"arn:aws:sagemaker:{self.region}:{self.account}:model-package/*/*",
                f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/*",
            ],
        )

    def models_full(self) -> iam.PolicyStatement:
        """Full CRUD access to SageMaker models and model packages."""
        read_statement = self.models_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "sagemaker:CreateModel",
                "sagemaker:CreateModelPackage",
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:DeleteModel",
                "sagemaker:DeleteModelPackage",
                "sagemaker:DeleteModelPackageGroup",
                "sagemaker:UpdateModelPackage",
                "sagemaker:UpdateModelPackageGroup",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=read_statement.resources,
        )

    def model_training(self) -> iam.PolicyStatement:
        """Create a policy statement for training SageMaker models.
        Returns:
            iam.PolicyStatement: The policy statement for SageMaker model training.
        """
        return iam.PolicyStatement(
            actions=["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:training-job/*"],
        )

    @staticmethod
    def model_training_logs() -> iam.PolicyStatement:
        """Create a policy statement for log interactions when training SageMaker models.
        Returns:
            iam.PolicyStatement: The policy statement for log interactions when training SageMaker models.
        """
        return iam.PolicyStatement(
            actions=[
                "logs:DescribeLogStreams",
                "logs:GetLogEvents",
                "logs:FilterLogEvents",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            resources=["arn:aws:logs:*:*:log-group:/aws/sagemaker/*", "arn:aws:logs:*:*:log-group:/aws/sagemaker/*:*"],
        )

    #####################
    #     Endpoints     #
    #####################
    @staticmethod
    def endpoint_discover() -> iam.PolicyStatement:
        """Discover SageMaker endpoints.
        Returns:
            iam.PolicyStatement: The policy statement for discovering SageMaker endpoints.
        """
        return iam.PolicyStatement(
            actions=["sagemaker:ListEndpoints"],
            resources=["*"],
        )

    def endpoint_read(self) -> iam.PolicyStatement:
        """Read SageMaker endpoints and configurations.
        Returns:
            iam.PolicyStatement: The policy statement for reading SageMaker endpoints.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:InvokeEndpoint",
                "sagemaker:ListTags",
            ],
            resources=[
                f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/*",
                f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint-config/*",
            ],
        )

    def endpoint_full(self) -> iam.PolicyStatement:
        """Full access to SageMaker endpoints and configurations.
        Returns:
            iam.PolicyStatement: The policy statement for full SageMaker endpoint access.
        """
        read_statement = self.endpoint_read()

        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "sagemaker:CreateEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=read_statement.resources,
        )

    ###########################
    #   Endpoint Monitoring   #
    ###########################
    @staticmethod
    def endpoint_monitoring_discovery() -> iam.PolicyStatement:
        """Create a policy statement for listing SageMaker endpoint monitoring resources.

        Returns:
            iam.PolicyStatement: The policy statement for listing operations.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListMonitoringSchedules",
                "sagemaker:ListMonitoringExecutions",
            ],
            resources=["*"],  # List operations require "*" resource
        )

    def endpoint_monitoring_schedules(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SageMaker endpoint monitoring schedules.

        Returns:
            iam.PolicyStatement: The policy statement for endpoint schedule operations.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:DescribeMonitoringSchedule",
                "sagemaker:DescribeMonitoringExecution",
                "sagemaker:CreateMonitoringSchedule",
                "sagemaker:UpdateMonitoringSchedule",
                "sagemaker:DeleteMonitoringSchedule",
                "sagemaker:StartMonitoringSchedule",
                "sagemaker:StopMonitoringSchedule",
                "sagemaker:ListTags",
            ],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:monitoring-schedule/*"],
        )

    def endpoint_data_quality(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SageMaker endpoint data quality jobs.

        Returns:
            iam.PolicyStatement: The policy statement for endpoint data quality monitoring operations.
        """
        return iam.PolicyStatement(
            actions=[
                # Data quality job definition operations
                "sagemaker:DescribeDataQualityJobDefinition",
                "sagemaker:CreateDataQualityJobDefinition",
                "sagemaker:UpdateDataQualityJobDefinition",
                "sagemaker:DeleteDataQualityJobDefinition",
                "sagemaker:ListDataQualityJobDefinitions",
            ],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:data-quality-job-definition/*"],
        )

    #####################
    #    EventBridge    #
    #####################
    def eventbridge_read(self) -> iam.PolicyStatement:
        """Policy for EventBridge read/discovery operations."""
        return iam.PolicyStatement(
            actions=[
                "events:DescribeEventBus",
            ],
            resources=[
                f"arn:aws:events:{self.region}:{self.account}:event-bus/workbench",
            ],
        )

    def eventbridge_write(self) -> iam.PolicyStatement:
        """Policy for EventBridge write operations."""
        return iam.PolicyStatement(
            actions=[
                "events:PutEvents",
            ],
            resources=[
                f"arn:aws:events:{self.region}:{self.account}:event-bus/workbench/*",
            ],
        )

    # For SNS notification operations
    def sns_notifications(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SNS notifications

        Returns:
            iam.PolicyStatement: The policy statement for SNS operations.
        """
        sns_resources = f"arn:aws:sns:{self.region}:{self.account}:*"

        return iam.PolicyStatement(
            actions=[
                "sns:CreateTopic",
                "sns:Subscribe",
                "sns:Publish",
            ],
            resources=[sns_resources],
        )

    @staticmethod
    def pipeline_list_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for listing SageMaker pipelines.

        Returns:
            iam.PolicyStatement: The policy statement for listing SageMaker pipelines.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListPipelines",
            ],
            resources=["*"],  # Broad permission necessary for listing operations
        )

    def pipeline_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for inspecting and running SageMaker Pipelines.

        Returns:
            iam.PolicyStatement: The policy statement for inspecting and running SageMaker Pipelines.
        """
        pipeline_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/*"
        execution_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline-execution/*"
        processing_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*"

        return iam.PolicyStatement(
            actions=[
                "sagemaker:DescribePipeline",
                "sagemaker:ListPipelineExecutions",
                "sagemaker:DescribePipelineExecution",
                "sagemaker:ListPipelineExecutionSteps",
                "sagemaker:StartPipelineExecution",
                # Actions for jobs
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob",
                "sagemaker:ListProcessingJobs",
                "sagemaker:StopProcessingJob",
                # Tagging
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=[
                pipeline_resources,
                execution_resources,
                processing_resources,
            ],
        )

    def ecr_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for pulling ECR Images.

        Returns:
            iam.PolicyStatement: The policy statement for pulling ECR Images.
        """
        return iam.PolicyStatement(
            actions=[
                # Actions for Pulling ECR Images
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",  # May be required for pulling layers
                # Public ECR actions
                "ecr-public:GetAuthorizationToken",
                "ecr-public:BatchCheckLayerAvailability",
                "ecr-public:BatchGetImage",
            ],
            resources=["*"],  # Typically, resources are set to "*" for ECR actions
        )

    ########################
    #    CloudWatch        #
    ########################
    @staticmethod
    def cloudwatch_metrics() -> iam.PolicyStatement:
        """CloudWatch metrics permissions.
        Returns:
            iam.PolicyStatement: The policy statement for CloudWatch metrics.
        """
        return iam.PolicyStatement(
            actions=[
                "cloudwatch:GetMetricData",
                "cloudwatch:PutMetricData",
            ],
            resources=["*"],  # CloudWatch metrics don't support specific resources
        )

    def cloudwatch_logs(self) -> iam.PolicyStatement:
        """CloudWatch logs permissions - CreateLogStream and PutLogEvents
        Returns:
            iam.PolicyStatement: The policy statement for the WorkbenchLogGroup.
        """
        return iam.PolicyStatement(
            actions=[
                "logs:CreateLogStream",  # Needed for dynamically creating log streams
                "logs:PutLogEvents",
            ],
            resources=[
                f"arn:aws:logs:{self.region}:{self.account}:log-group:WorkbenchLogGroup",
                f"arn:aws:logs:{self.region}:{self.account}:log-group:WorkbenchLogGroup:*",
            ],
        )

    # For CloudWatch alarm operations
    def cloudwatch_alarms(self) -> iam.PolicyStatement:
        """Create a policy statement for managing CloudWatch alarms.

        Returns:
            iam.PolicyStatement: The policy statement for CloudWatch alarms.
        """
        return iam.PolicyStatement(
            actions=[
                "cloudwatch:PutMetricAlarm",
                "cloudwatch:DescribeAlarms",
                "cloudwatch:DeleteAlarms",
            ],
            resources=[f"arn:aws:cloudwatch:{self.region}:{self.account}:alarm:*"],
        )

    ##########################
    #   Workbench Dashboard  #
    ##########################
    def dashboard_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for additional permissions needed by Workbench Dashboard.

        Returns:
            iam.PolicyStatement: The policy statement needed by Workbench Dashboard.
        """
        return iam.PolicyStatement(
            actions=[
                # ECS action
                "ecs:DescribeServices",
                # ELB action
                "elasticloadbalancing:DescribeLoadBalancers",
            ],
            resources=["*"],
        )

    #####################
    #  Parameter Store  #
    #####################
    @staticmethod
    def parameter_store_discover() -> iam.PolicyStatement:
        """Discover Parameter Store parameters.
        Returns:
            iam.PolicyStatement: The policy statement for discovering Parameter Store parameters.
        """
        return iam.PolicyStatement(
            actions=["ssm:DescribeParameters"],
            resources=["*"],
        )

    def parameter_store_read(self) -> iam.PolicyStatement:
        """Read Parameter Store parameters.
        Returns:
            iam.PolicyStatement: The policy statement for reading Parameter Store parameters.
        """
        return iam.PolicyStatement(
            actions=[
                "ssm:GetParameter",
                "ssm:GetParameters",
                "ssm:GetParametersByPath",
            ],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter/*",
            ],
        )

    def parameter_store_full(self) -> iam.PolicyStatement:
        """Full access to Parameter Store parameters.
        Returns:
            iam.PolicyStatement: The policy statement for full Parameter Store access.
        """
        read_statement = self.parameter_store_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "ssm:PutParameter",
                "ssm:DeleteParameter",
            ],
            resources=read_statement.resources,
        )

    def workbench_datasource_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench DataSources"""
        policy_statements = [
            self.s3_full(),
            self.s3_public(),
            self.glue_catalog_full(),
            self.glue_databases_full(),
            self.athena_read(),
            self.cloudwatch_logs(),
            self.parameter_store_full(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchDataSourcePolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchDataSourcePolicy",
        )

    def workbench_featureset_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench FeatureSets"""
        policy_statements = [
            self.s3_full(),
            self.glue_catalog_full(),
            self.glue_databases_full(),
            self.athena_read(),
            self.featurestore_discovery(),
            self.featurestore_full(),
            self.cloudwatch_logs(),
            self.parameter_store_full(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchFeatureSetPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchFeatureSetPolicy",
        )

    """In SageMaker, certain operations, such as creating training jobs, endpoint deployments, or batch transform jobs,
       require SageMaker to assume an IAM role. This role provides SageMaker with permissions to access AWS resources 
       on your behalf, such as reading training data from S3, writing model artifacts, or logging to CloudWatch."""
    def sagemaker_pass_role_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for SageMaker to assume the Execution Role"""
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=["arn:aws:iam::*:role/*"],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
        )

    def workbench_model_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.models_discovery(),
            self.models_full(),
            self.model_training(),
            self.model_training_logs(),
            self.ecr_policy_statement(),
            self.cloudwatch_metrics(),
            self.cloudwatch_logs(),
            self.sagemaker_pass_role_policy_statement(),
            self.parameter_store_full(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchModelPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchModelPolicy",
        )

    def workbench_endpoint_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.endpoint_discover(),
            self.endpoint_full(),
            self.endpoint_data_quality(),
            self.endpoint_monitoring_discovery(),
            self.endpoint_monitoring_schedules(),
            self.cloudwatch_metrics(),
            self.cloudwatch_logs(),
            self.parameter_store_full(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchEndpointPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchEndpointPolicy",
        )

    def workbench_pipeline_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Pipelines"""
        policy_statements = [
            self.pipeline_list_policy_statement(),
            self.pipeline_policy_statement(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchPipelinePolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchPipelinePolicy",
        )

    def create_api_execution_role(self) -> iam.Role:
        """Create the Workbench Execution Role for API-related tasks"""

        # Define the base assumed by principals with ECS service principal
        assumed_by = iam.CompositePrincipal(
            iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            iam.ServicePrincipal("sagemaker.amazonaws.com"),
            iam.ServicePrincipal("glue.amazonaws.com"),
        )

        # If sso_group is provided, add the condition to the trust relationship
        if self.sso_group:
            sso_group_arn_1 = (
                f"arn:aws:iam::{self.account}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_{self.sso_group}_*"
            )
            sso_group_arn_2 = (
                f"arn:aws:iam::{self.account}:role/aws-reserved/sso.amazonaws.com/*/AWSReservedSSO_{self.sso_group}_*"
            )
            condition = {"ArnLike": {"aws:PrincipalArn": [sso_group_arn_1, sso_group_arn_2]}}
            assumed_by.add_principals(iam.AccountPrincipal(self.account).with_conditions(condition))
        else:
            assumed_by.add_principals(iam.AccountPrincipal(self.account))

        # Create the role with the trust relationships
        api_execution_role = iam.Role(
            self,
            id=self.workbench_role_name,
            assumed_by=assumed_by,
            role_name=self.workbench_role_name,
        )

        # Create and attach the Workbench managed policies to the role
        api_execution_role.add_to_policy(self.glue_pass_role())
        api_execution_role.add_to_policy(self.glue_jobs_discover())
        api_execution_role.add_to_policy(self.glue_jobs_full())
        api_execution_role.add_to_policy(self.parameter_store_discover())
        api_execution_role.add_to_policy(self.parameter_store_full())
        api_execution_role.add_managed_policy(self.datasource_policy)
        api_execution_role.add_managed_policy(self.featureset_policy)
        api_execution_role.add_managed_policy(self.model_policy)
        api_execution_role.add_managed_policy(self.endpoint_policy)
        api_execution_role.add_managed_policy(self.pipeline_policy)

        return api_execution_role

    def create_lambda_role(self) -> iam.Role:
        """Create the Workbench Lambda Role."""
        lambda_assumed_by = iam.ServicePrincipal("lambda.amazonaws.com")
        lambda_role = iam.Role(
            self,
            id="Workbench-LambdaRole",
            assumed_by=lambda_assumed_by,
            role_name="Workbench-LambdaRole",
        )

        # Add a subset of policies for the Lambda Role
        lambda_role.add_to_policy(self.parameter_store_full())
        lambda_role.add_managed_policy(self.datasource_policy)
        lambda_role.add_managed_policy(self.featureset_policy)
        lambda_role.add_managed_policy(self.model_policy)
        lambda_role.add_managed_policy(self.endpoint_policy)
        lambda_role.add_managed_policy(self.pipeline_policy)
        return lambda_role

    def create_glue_role(self) -> iam.Role:
        """Create the Workbench Glue Role."""
        glue_assumed_by = iam.ServicePrincipal("glue.amazonaws.com")
        glue_role = iam.Role(
            self,
            id="Workbench-GlueRole",
            assumed_by=glue_assumed_by,
            role_name="Workbench-GlueRole",
        )

        # Add a subset of policies for the Glue Role
        glue_role.add_to_policy(self.parameter_store_full())
        glue_role.add_managed_policy(self.datasource_policy)
        glue_role.add_managed_policy(self.featureset_policy)
        glue_role.add_managed_policy(self.model_policy)
        glue_role.add_managed_policy(self.endpoint_policy)
        glue_role.add_managed_policy(self.pipeline_policy)
        return glue_role
