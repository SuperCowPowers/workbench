from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
)
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
            actions=read_statement.actions + [
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

    def glue_pass_role(self) -> iam.PolicyStatement:
        """Allows us to specify the Workbench-Glue role when creating a Glue Job"""
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[f"arn:aws:iam::{self.account}:role/Workbench-GlueRole"],
            conditions={"StringEquals": {"iam:PassedToService": "glue.amazonaws.com"}},
        )

    def glue_job_read_policy(self) -> iam.PolicyStatement:
        """Policy for read-only Glue job actions."""
        return iam.PolicyStatement(
            actions=[
                "glue:GetJobs",
                "glue:GetJob",
                "glue:GetJobRun",
                "glue:GetJobRuns",
            ],
            resources=[
                "*",  # Needed for GetJobs
            ],
        )

    def glue_job_create_policy(self) -> iam.PolicyStatement:
        """Policy for create/update Glue jobs and triggers."""
        return iam.PolicyStatement(
            actions=[
                "glue:CreateJob",
                "glue:UpdateJob",
                "glue:StartJobRun",
                "glue:CreateTrigger",
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:job/*",
                f"arn:aws:glue:{self.region}:{self.account}:trigger/*",
            ],
        )

    @staticmethod
    def glue_job_connections_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for Glue job network and connection permissions."""
        return iam.PolicyStatement(
            actions=[
                # Glue connection actions
                "glue:GetConnection",
                "glue:GetConnections",
                # EC2 network actions for VPC access and ENI management
                "ec2:Describe*",
                "ec2:CreateNetworkInterface",
                "ec2:DeleteNetworkInterface",
                "ec2:DescribeNetworkInterfaces",
                "ec2:CreateTags",
            ],
            resources=["*"],  # Broad permissions for Glue connections and VPC network queries
        )

    def glue_catalog_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for Glue Data Catalog-wide permissions."""
        catalog_arn = f"arn:aws:glue:{self.region}:{self.account}:catalog"

        return iam.PolicyStatement(
            actions=[
                # Catalog-wide permissions
                "glue:GetDatabases",
                "glue:GetDatabase",
                "glue:CreateDatabase",
                "glue:SearchTables",
                "glue:GetTables",
                "glue:GetTable",
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
                "glue:GetPartitions",
            ],
            resources=[catalog_arn],
        )

    def glue_database_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for specific Glue databases and tables."""
        # ARNs for the databases and tables
        workbench_database_arn = f"arn:aws:glue:{self.region}:{self.account}:database/workbench"
        workbench_table_arn = f"arn:aws:glue:{self.region}:{self.account}:table/workbench/*"
        sagemaker_featurestore_database_arn = (
            f"arn:aws:glue:{self.region}:{self.account}:database/sagemaker_featurestore"
        )
        sagemaker_table_arn = f"arn:aws:glue:{self.region}:{self.account}:table/sagemaker_featurestore/*"
        inf_store_database_arn = f"arn:aws:glue:{self.region}:{self.account}:database/inference_store"
        inf_store_table_arn = f"arn:aws:glue:{self.region}:{self.account}:table/inference_store/*"

        return iam.PolicyStatement(
            actions=[
                # Database-specific permissions
                "glue:GetDatabase",
                "glue:GetTable",
                "glue:GetTables",
                "glue:UpdateTable",
                "glue:CreateTable",
                "glue:DeleteTable",
                # Partition-specific actions
                "glue:GetPartition",
                "glue:GetPartitions",
            ],
            resources=[
                workbench_database_arn,
                workbench_table_arn,
                sagemaker_featurestore_database_arn,
                sagemaker_table_arn,
                inf_store_database_arn,
                inf_store_table_arn,
            ],
        )

    def eventbridge_policy(self) -> iam.PolicyStatement:
        """Policy for EventBridge events."""
        return iam.PolicyStatement(
            actions=[
                "events:PutEvents",
                "events:DescribeEventBus",
            ],
            resources=[
                f"arn:aws:events:{self.region}:{self.account}:event-bus/workbench",
            ],
        )

    def athena_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for Athena actions that involve S3 buckets.

        Returns:
            iam.PolicyStatement: The policy statement for Athena S3 bucket access.
        """
        return iam.PolicyStatement(
            actions=[
                "athena:ListQueryExecutions",
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:StopQueryExecution",
            ],
            resources=["*"],  # Athena Actions are not resource-specific in IAM policies
        )

    @staticmethod
    def athena_workgroup_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for Athena WorkGroup actions.

        Returns:
            iam.PolicyStatement: The policy statement for Athena WorkGroup access.
        """
        return iam.PolicyStatement(
            actions=[
                "athena:GetWorkGroup",
                "athena:ListWorkGroups",
            ],
            resources=["*"],  # Listing WorkGroups not S3 bucket specific
        )

    @staticmethod
    def featurestore_list_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for listing SageMaker feature groups.

        Returns:
            iam.PolicyStatement: The policy statement allowing listing of SageMaker feature groups.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListFeatureGroups",  # Action for listing feature groups
            ],
            resources=["*"],  # Broad permission necessary for listing operations
        )

    def featurestore_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for broad SageMaker Feature Store access using self attributes for region and account.

        Returns:
            iam.PolicyStatement: The policy statement for broad SageMaker Feature Store access.
        """
        # Define the SageMaker Feature Store resources
        resources = [f"arn:aws:sagemaker:{self.region}:{self.account}:feature-group/*"]

        return iam.PolicyStatement(
            actions=[
                # Define the SageMaker Feature Store actions you need
                "sagemaker:CreateFeatureGroup",
                "sagemaker:DeleteFeatureGroup",
                "sagemaker:DescribeFeatureGroup",
                "sagemaker:GetRecord",
                "sagemaker:PutRecord",
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=resources,
        )

    @staticmethod
    def model_list_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for listing SageMaker models.

        Returns:
            iam.PolicyStatement: The policy statement for listing SageMaker models.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListModelPackageGroups",
                "sagemaker:ListModelPackages",
                "sagemaker:ListModels",
            ],
            resources=["*"],  # Broad permission necessary for listing operations
        )

    def model_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for accessing SageMaker model package groups and model packages.

        Returns:
            iam.PolicyStatement: The policy statement for SageMaker model resources access.
        """
        # Define the SageMaker Model Package Group and Model Package ARNs
        model_package_group_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/*"
        model_package_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:model-package/*/*"
        model_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:model/*"

        # Sagemaker Pipelines
        processing_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*"

        return iam.PolicyStatement(
            actions=[
                # Actions for model package groups
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:DeleteModelPackageGroup",
                "sagemaker:DescribeModelPackageGroup",
                "sagemaker:GetModelPackageGroup",
                "sagemaker:UpdateModelPackageGroup",
                # Actions for model packages
                "sagemaker:CreateModelPackage",
                "sagemaker:DeleteModelPackage",
                "sagemaker:DescribeModelPackage",
                "sagemaker:GetModelPackage",
                "sagemaker:UpdateModelPackage",
                # Actions for models
                "sagemaker:CreateModel",
                "sagemaker:DeleteModel",
                "sagemaker:DescribeModel",
                # Additional actions
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=[model_package_group_arn, model_package_arn, model_arn, processing_arn],
        )

    def model_training_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for training SageMaker models.

        Returns:
            iam.PolicyStatement: The policy statement for SageMaker model training.
        """

        return iam.PolicyStatement(
            actions=["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:training-job/*"],
        )

    @staticmethod
    def model_training_log_statement() -> iam.PolicyStatement:
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
            resources=["*"],  # Broad permission necessary for log operations
        )

    @staticmethod
    def endpoint_list_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for listing SageMaker endpoints.

        Returns:
            iam.PolicyStatement: The policy statement for listing SageMaker endpoints.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListEndpoints",
            ],
            resources=["*"],  # Broad permission necessary for listing operations
        )

    def endpoint_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for accessing SageMaker endpoints.

        Returns:
            iam.PolicyStatement: The policy statement for SageMaker endpoint access.
        """
        # Define the SageMaker Endpoint ARN
        endpoint_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/*"
        endpoint_config_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint-config/*"

        return iam.PolicyStatement(
            actions=[
                # Actions for endpoints
                "sagemaker:CreateEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:InvokeEndpoint",
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=[
                endpoint_arn,
                endpoint_config_arn,
            ],
        )

    # Listing monitoring schedules and executions
    def monitoring_list_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for listing SageMaker monitoring resources.

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

    # For specific monitoring schedule operations
    def monitoring_schedule_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SageMaker monitoring schedules.

        Returns:
            iam.PolicyStatement: The policy statement for monitoring schedule operations.
        """
        schedule_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:monitoring-schedule/*"
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
            resources=[schedule_resources],
        )

    def data_quality_job_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SageMaker data quality job definitions and monitoring.

        Returns:
            iam.PolicyStatement: The policy statement for data quality monitoring operations.
        """
        # Define resources for data quality job definitions
        job_definition_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:data-quality-job-definition/*"

        return iam.PolicyStatement(
            actions=[
                # Data quality job definition operations
                "sagemaker:DescribeDataQualityJobDefinition",
                "sagemaker:CreateDataQualityJobDefinition",
                "sagemaker:UpdateDataQualityJobDefinition",
                "sagemaker:DeleteDataQualityJobDefinition",
                "sagemaker:ListDataQualityJobDefinitions",
            ],
            resources=[job_definition_resources],
        )

    # For data capture operations
    def data_capture_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SageMaker endpoint data capture.

        Returns:
            iam.PolicyStatement: The policy statement for endpoint data capture operations.
        """
        endpoint_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/*"
        endpoint_config_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint-config/*"

        return iam.PolicyStatement(
            actions=[
                "sagemaker:EnableCapture",
                "sagemaker:DisableCapture",
            ],
            resources=[
                endpoint_resources,
                endpoint_config_resources,
            ],
        )

    # For CloudWatch alarm operations
    def monitoring_cloudwatch_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for managing CloudWatch alarms for monitoring.

        Returns:
            iam.PolicyStatement: The policy statement for CloudWatch operations.
        """
        cloudwatch_resources = f"arn:aws:cloudwatch:{self.region}:{self.account}:alarm:*"

        return iam.PolicyStatement(
            actions=[
                "cloudwatch:PutMetricAlarm",
                "cloudwatch:DescribeAlarms",
                "cloudwatch:DeleteAlarms",
            ],
            resources=[cloudwatch_resources],
        )

    # For SNS notification operations
    def monitoring_sns_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for managing SNS notifications for monitoring.

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

    # Helper method to get all monitoring policy statements
    def all_monitoring_policy_statements(self) -> List[iam.PolicyStatement]:
        """Get all policy statements needed for SageMaker endpoint monitoring.

        Returns:
            List[iam.PolicyStatement]: A list of all policy statements for monitoring.
        """
        return [
            self.monitoring_list_policy_statement(),
            self.monitoring_schedule_policy_statement(),
            self.data_capture_policy_statement(),
            self.monitoring_cloudwatch_policy_statement(),
            self.monitoring_sns_policy_statement(),
            self.data_quality_job_policy_statement(),
        ]

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

    @staticmethod
    def cloudwatch_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for accessing CloudWatch metric data.

        Returns:
            iam.PolicyStatement: The policy statement for CloudWatch GetMetricData access.
        """
        return iam.PolicyStatement(
            actions=[
                "cloudwatch:GetMetricData",
                "cloudwatch:PutMetricData",
            ],
            resources=["*"],  # Cloudwatch does not support specific resources
        )

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

    def parameter_store_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for accessing AWS Systems Manager Parameter Store.

        Returns:
            iam.PolicyStatement: The policy statement for accessing AWS Systems Manager Parameter Store.
        """
        return iam.PolicyStatement(
            actions=[
                "ssm:DescribeParameters",
                "ssm:GetParameter",
                "ssm:GetParameters",
                "ssm:GetParametersByPath",
                "ssm:PutParameter",
                "ssm:DeleteParameter",
            ],
            resources=["*"],  # Broad permission necessary for Parameter Store operations
        )

    def workbench_datasource_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench DataSources"""
        policy_statements = [
            self.s3_full(),
            self.s3_public(),
            self.glue_job_read_policy(),
            self.glue_job_create_policy(),
            self.glue_job_connections_policy_statement(),
            self.glue_catalog_policy_statement(),
            self.glue_database_policy_statement(),
            self.athena_policy_statement(),
            self.athena_workgroup_policy_statement(),
            self.parameter_store_policy_statement(),
            self.eventbridge_policy(),
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
            self.glue_catalog_policy_statement(),
            self.glue_database_policy_statement(),
            self.athena_policy_statement(),
            self.athena_workgroup_policy_statement(),
            self.featurestore_list_policy_statement(),
            self.featurestore_policy_statement(),
            self.parameter_store_policy_statement(),
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
        """Create a policy statement for SageMaker to assume the Execution Role

        Args:
            workbench_api_role (iam.Role): The Workbench Execution Role
        """
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=["arn:aws:iam::*:role/*"],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
        )

    def workbench_model_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.model_list_policy_statement(),
            self.model_policy_statement(),
            self.model_training_statement(),
            self.model_training_log_statement(),
            self.ecr_policy_statement(),
            self.cloudwatch_policy_statement(),
            self.sagemaker_pass_role_policy_statement(),
            self.parameter_store_policy_statement(),
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
            self.endpoint_list_policy_statement(),
            self.endpoint_policy_statement(),
            self.cloudwatch_policy_statement(),
            self.parameter_store_policy_statement(),
        ]
        # Add the monitoring policy statements to the endpoint policy
        policy_statements += self.all_monitoring_policy_statements()
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
        glue_role.add_managed_policy(self.datasource_policy)
        glue_role.add_managed_policy(self.featureset_policy)
        glue_role.add_managed_policy(self.model_policy)
        glue_role.add_managed_policy(self.endpoint_policy)
        glue_role.add_managed_policy(self.pipeline_policy)
        return glue_role
