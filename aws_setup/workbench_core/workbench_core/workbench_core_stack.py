from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
    aws_logs as logs,
    aws_athena as athena,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct
from typing import Any, List
from dataclasses import dataclass, field


@dataclass
class WorkbenchCoreStackProps:
    workbench_bucket: str
    sso_groups: List[str] = field(default_factory=list)
    additional_buckets: List[str] = field(default_factory=list)
    existing_vpc_id: str = None
    subnet_ids: List[str] = field(default_factory=list)  # Optional subnets ids for the Batch compute environment


class WorkbenchCoreStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        props: WorkbenchCoreStackProps,
        **kwargs: Any,
    ) -> None:
        desc = "Workbench Core: Defines the core Workbench resources, roles, and policies for the Workbench API."
        super().__init__(scope, construct_id, env=env, description=desc, **kwargs)

        # Print the environment details
        print("Environment")
        print(env)

        # Grab our properties
        self.workbench_bucket = props.workbench_bucket
        self.sso_groups = props.sso_groups
        self.additional_buckets = props.additional_buckets
        self.existing_vpc_id = props.existing_vpc_id
        self.subnet_ids = props.subnet_ids

        # Workbench Role Names
        self.execution_role_name = "Workbench-ExecutionRole"  # Main role
        self.readonly_role_name = "Workbench-ReadOnlyRole"  # Read only operations
        self.glue_role_name = "Workbench-GlueRole"
        self.lambda_role_name = "Workbench-LambdaRole"
        self.batch_role_name = "Workbench-BatchRole"

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

        # Add our Athena workgroup
        self.athena_workgroup = self.create_athena_workgroup()

        # Create our managed polices
        self.s3_read_policy = self.workbench_s3_read_policy()
        self.s3_full_policy = self.workbench_s3_full_policy()
        self.glue_connections_policy = self.workbench_glue_connections_policy()
        self.datasource_read_policy = self.workbench_datasource_read_policy()
        self.datasource_policy = self.workbench_datasource_policy()
        self.featureset_read_policy = self.workbench_featureset_read_policy()
        self.featureset_policy = self.workbench_featureset_policy()
        self.model_read_policy = self.workbench_model_read_policy()
        self.model_policy = self.workbench_model_policy()
        self.endpoint_read_policy = self.workbench_endpoint_read_policy()
        self.endpoint_policy = self.workbench_endpoint_policy()
        self.dataframe_store_read_policy = self.workbench_dataframe_store_read_policy()
        self.dataframe_store_full_policy = self.workbench_dataframe_store_full_policy()
        self.parameter_store_read_policy = self.workbench_parameter_store_read_policy()
        self.parameter_store_full_policy = self.workbench_parameter_store_full_policy()
        self.inference_store_read_policy = self.workbench_inference_store_read_policy()
        self.inference_store_full_policy = self.workbench_inference_store_full_policy()

        # Create our main Workbench API Execution Role and Read Only Role
        self.workbench_execution_role = self.create_execution_role()
        self.workbench_readonly_role = self.create_readonly_role()
        self._create_sso_instructions(self.workbench_execution_role, self.workbench_readonly_role)

        # Create additional roles for Lambda, Glue, and Batch
        self.workbench_lambda_role = self.create_lambda_role()
        self.workbench_glue_role = self.create_glue_role()
        self.workbench_batch_role = self.create_batch_role()

        # Export role ARNs that might  be used by other stacks (like WorkbenchCompute)
        CfnOutput(
            self,
            "LambdaRoleArn",
            value=self.workbench_lambda_role.role_arn,
            export_name=f"{self.stack_name}-LambdaRoleArn",
        )
        CfnOutput(
            self, "GlueRoleArn", value=self.workbench_glue_role.role_arn, export_name=f"{self.stack_name}-GlueRoleArn"
        )
        CfnOutput(
            self,
            "BatchRoleArn",
            value=self.workbench_batch_role.role_arn,
            export_name=f"{self.stack_name}-BatchRoleArn",
        )

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
            resources=self._workbench_database_arns() + self._inference_database_arns(),
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
            resources=self._workbench_database_arns() + self._inference_database_arns(),
        )

    def _workbench_database_arns(self) -> list[str]:
        """Helper to get Workbench-managed database/table ARNs."""
        return [
            f"arn:aws:glue:{self.region}:{self.account}:database/workbench",
            f"arn:aws:glue:{self.region}:{self.account}:table/workbench/*",
            f"arn:aws:glue:{self.region}:{self.account}:database/sagemaker_featurestore",
            f"arn:aws:glue:{self.region}:{self.account}:table/sagemaker_featurestore/*",
        ]

    def _inference_database_arns(self) -> list[str]:
        """Helper to get inference store database/table ARNs."""
        return [
            f"arn:aws:glue:{self.region}:{self.account}:database/inference_store",
            f"arn:aws:glue:{self.region}:{self.account}:table/inference_store/*",
        ]

    #####################
    #     Glue Jobs     #
    #####################
    def glue_jobs_s3_read(self) -> iam.PolicyStatement:
        """S3 Read for Glue Jobs default script location"""
        return iam.PolicyStatement(
            actions=["s3:GetObject", "s3:GetBucketLocation", "s3:ListBucket"],
            resources=[
                f"arn:aws:s3:::aws-glue-assets-{self.account}-{self.region}",
                f"arn:aws:s3:::aws-glue-assets-{self.account}-{self.region}/*",
            ],
        )

    def glue_job_logs(self) -> iam.PolicyStatement:
        """Create a policy statement for Glue job CloudWatch logs.

        Returns:
            iam.PolicyStatement: The policy statement for Glue job log interactions.
        """
        return iam.PolicyStatement(
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            resources=[
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws-glue/*",
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws-glue/*:*",
            ],
        )

    def glue_connections(self) -> iam.PolicyStatement:
        """Create a policy statement for Glue connection access.

        Returns:
            iam.PolicyStatement: The policy statement for Glue connection interactions.
        """
        return iam.PolicyStatement(
            actions=[
                "glue:GetConnection",
            ],
            resources=[
                # GetConnection requires access to the catalog resource since the
                # Glue Catalog is used for metadata storage and connection retrieval
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:connection/*",
            ],
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

    def glue_pass_role(self) -> iam.PolicyStatement:
        """Allows us to specify the Workbench-Glue role when creating a Glue Job"""
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[f"arn:aws:iam::{self.account}:role/{self.glue_role_name}"],
            conditions={"StringEquals": {"iam:PassedToService": "glue.amazonaws.com"}},
        )

    ##################
    #   Batch Jobs   #
    ##################
    def batch_job_logs(self) -> iam.PolicyStatement:
        """Create a policy statement for Glue job CloudWatch logs.

        Returns:
            iam.PolicyStatement: The policy statement for Glue job log interactions.
        """
        return iam.PolicyStatement(
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            resources=[
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/batch/job",
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/batch/job:*",
            ],
        )

    @staticmethod
    def batch_jobs_discover() -> iam.PolicyStatement:
        """Discovery actions that absolutely require * resource."""
        return iam.PolicyStatement(
            actions=[
                "batch:ListJobs",  # Requires * to list across all queues
                "batch:DescribeJobs",  # Requires * even for specific job IDs (AWS API design)
            ],
            resources=["*"],
        )

    def batch_jobs_read(self) -> iam.PolicyStatement:
        """Read-only access to specific Batch resources."""
        return iam.PolicyStatement(
            actions=[
                "batch:DescribeJobDefinitions",
                "batch:DescribeJobQueues",
                "batch:DescribeComputeEnvironments",
            ],
            resources=[
                f"arn:aws:batch:{self.region}:{self.account}:job-definition/workbench-*",
                f"arn:aws:batch:{self.region}:{self.account}:job-queue/workbench-*",
                f"arn:aws:batch:{self.region}:{self.account}:compute-environment/workbench-*",
                f"arn:aws:batch:{self.region}:{self.account}:job/*",  # Job IDs are random UUIDs
            ],
        )

    def batch_jobs_full(self) -> iam.PolicyStatement:
        """Full access to specific Batch resources."""
        read_statement = self.batch_jobs_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "batch:RegisterJobDefinition",
                "batch:DeregisterJobDefinition",
                "batch:SubmitJob",
                "batch:TerminateJob",
                "batch:CancelJob",
            ],
            resources=read_statement.resources,
        )

    def batch_pass_role(self) -> iam.PolicyStatement:
        """Allows us to specify the Workbench-Batch role when creating a Batch Job"""
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[f"arn:aws:iam::{self.account}:role/{self.batch_role_name}"],
            conditions={"StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}},
        )

    ####################
    #    SQS Queues    #
    ####################
    @staticmethod
    def sqs_discover() -> iam.PolicyStatement:
        """Discovery - list all SQS queues."""
        return iam.PolicyStatement(
            actions=["sqs:ListQueues"],
            resources=["*"],  # List operations require "*" resource
        )

    def sqs_read(self) -> iam.PolicyStatement:
        """Read-only access to SQS queues."""
        return iam.PolicyStatement(
            actions=[
                "sqs:GetQueueAttributes",
                "sqs:ReceiveMessage",
                "sqs:DeleteMessage",
                "sqs:GetQueueUrl",
            ],
            resources=[f"arn:aws:sqs:{self.region}:{self.account}:workbench-*"],
        )

    def sqs_full(self) -> iam.PolicyStatement:
        """Full access to SQS queues."""
        read_statement = self.sqs_read()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "sqs:SendMessage",
                "sqs:CreateQueue",
                "sqs:PurgeQueue",
            ],
            resources=read_statement.resources,
        )

    #####################
    #  DataFrame Store  #
    #####################
    def dataframe_store_read(self) -> iam.PolicyStatement:
        """Create a policy statement for reading from the DataFrame Store.

        Returns:
            iam.PolicyStatement: The policy statement for reading from the DataFrame Store.
        """

        # Just return the S3 read policy for the DataFrame Store
        return self.s3_read()

    def dataframe_store_full(self) -> iam.PolicyStatement:
        """Create a policy statement for full access to the DataFrame Store.

        Returns:
            iam.PolicyStatement: The policy statement for full access to the DataFrame Store.
        """

        # Just return the S3 full policy for the DataFrame Store
        return self.s3_full()

    #####################
    #  Inference Store  #
    #####################
    def glue_database_read_just_inference_store(self) -> iam.PolicyStatement:
        """Create a policy statement for reading from the Parameter Store.

        Note: This is basically glue_databases_read but scoped to just the inference_store database.

        Returns:
            iam.PolicyStatement: The policy statement for reading from the Inference Store.
        """
        return iam.PolicyStatement(
            actions=[
                "glue:GetDatabase",
                "glue:GetTable",
                "glue:GetTables",
                "glue:GetPartition",
                "glue:GetPartitions",
            ],
            resources=self._inference_database_arns(),
        )

    def glue_database_full_just_inference_store(self) -> iam.PolicyStatement:
        """Create a policy statement for full access to the Inference Store.

        Note: This is basically glue_databases_full but scoped to just the inference_store database.

        Returns:
            iam.PolicyStatement: The policy statement for full access to the Inference Store.
        """
        read_statement = self.glue_database_read_just_inference_store()
        return iam.PolicyStatement(
            actions=read_statement.actions
            + [
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
            ],
            resources=read_statement.resources,
        )

    def s3_full_just_inference_store(self) -> iam.PolicyStatement:
        """S3 write permissions for inference store data only."""
        return iam.PolicyStatement(
            actions=[
                "s3:PutObject",  # Write inference store data
                "s3:DeleteObject",  # Delete inference store data
            ],
            resources=[
                f"arn:aws:s3:::{self.workbench_bucket}/athena/inference_store/inference_store/*",
            ],
        )

    ####################
    #  VPC/Networking  #
    ####################
    @staticmethod
    def vpc_discovery() -> iam.PolicyStatement:
        """Create a policy statement for VPC resource discovery.

        Returns:
            iam.PolicyStatement: The policy statement for VPC subnet and security group discovery operations.
        """
        return iam.PolicyStatement(
            actions=[
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeVpcEndpoints",
                "ec2:DescribeRouteTables",
            ],
            resources=[
                "*",  # EC2 describe operations require wildcard resources
            ],
        )

    def vpc_network_interface_management(self) -> iam.PolicyStatement:
        """Create a policy statement for VPC network interface management.
        Returns:
            iam.PolicyStatement: The policy statement for ENI creation and discovery in VPC.
        """
        return iam.PolicyStatement(
            actions=[
                "ec2:CreateNetworkInterface",
                "ec2:DescribeNetworkInterfaces",
                "ec2:CreateTags",
            ],
            resources=[
                f"arn:aws:ec2:{self.region}:{self.account}:network-interface/*",
                f"arn:aws:ec2:{self.region}:{self.account}:subnet/*",
                f"arn:aws:ec2:{self.region}:{self.account}:security-group/*",
            ],
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

    def athena_query_results_s3(self) -> iam.PolicyStatement:
        """S3 permissions for Athena query results bucket."""
        return iam.PolicyStatement(
            actions=[
                "s3:PutObject",  # Required: Athena writes query results even for SELECT queries
                "s3:GetObject",  # Needed to retrieve results later
                "s3:ListBucket",  # Needed for bucket operations
            ],
            resources=[
                f"arn:aws:s3:::aws-athena-query-results-{self.account}-{self.region}",
                f"arn:aws:s3:::aws-athena-query-results-{self.account}-{self.region}/*",
            ],
        )

    # Create our Athena workgroup
    # Note: We don't create the Athena results bucket here, just the workgroup
    # The bucket will be created as part of th aws_account_setup.py script
    def create_athena_workgroup(self) -> athena.CfnWorkGroup:
        """Create workbench-specific Athena workgroup with S3 output location."""
        athena_results_bucket = f"aws-athena-query-results-{self.account}-{self.region}"

        return athena.CfnWorkGroup(
            self,
            "WorkbenchAthenaWorkGroup",
            name="workbench-workgroup",
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{athena_results_bucket}/workbench/"
                )
            ),
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

    def model_training_logs(self) -> iam.PolicyStatement:
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
            resources=[
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*:*",
            ],
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

    def endpoint_monitoring_processing(self) -> iam.PolicyStatement:
        """Processing jobs for Model Monitor baselines and executions."""
        return iam.PolicyStatement(
            actions=[
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob",
                "sagemaker:StopProcessingJob",
            ],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*"],
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
    def ecr_policy_statement() -> iam.PolicyStatement:
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

    def cloudwatch_monitor(self) -> iam.PolicyStatement:
        """CloudWatch logs monitoring permissions - read and describe operations
        Returns:
            iam.PolicyStatement: The policy statement for monitoring WorkbenchLogGroup.
        """
        return iam.PolicyStatement(
            actions=[
                "logs:DescribeLogStreams",
                "logs:GetLogEvents",
                "logs:FilterLogEvents",
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
    @staticmethod
    def dashboard_policy_statement() -> iam.PolicyStatement:
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

    ##################################
    #   Workbench Managed Policies   #
    ##################################
    def workbench_s3_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench S3 Read-Only access"""
        policy_statements = [
            self.s3_read(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchS3ReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchS3ReadPolicy",
        )

    def workbench_s3_full_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench S3 Full access"""
        policy_statements = [
            self.s3_full(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchS3FullPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchS3FullPolicy",
        )

    def workbench_glue_connections_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Glue Connections"""
        policy_statements = [
            self.glue_job_logs(),
            self.glue_connections(),
            self.vpc_discovery(),
            self.vpc_network_interface_management(),
            self.cloudwatch_logs(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchGlueConnectionsPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchGlueConnectionsPolicy",
        )

    def workbench_datasource_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench DataSources (READ-ONLY)"""
        policy_statements = [
            self.s3_read(),
            self.s3_public(),
            self.glue_catalog_read(),
            self.glue_databases_read(),
            self.athena_read(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchDataSourceReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchDataSourceReadPolicy",
        )

    def workbench_datasource_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench DataSources (FULL)"""
        policy_statements = [
            self.s3_full(),
            self.s3_public(),
            self.glue_catalog_full(),
            self.glue_databases_full(),
            self.athena_read(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_full(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchDataSourcePolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchDataSourcePolicy",
        )

    def workbench_featureset_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench FeatureSets (READ-ONLY)"""
        policy_statements = [
            self.s3_full(),
            self.glue_catalog_full(),
            self.glue_databases_full(),
            self.athena_read(),
            self.featurestore_discovery(),
            self.featurestore_full(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchFeatureSetReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchFeatureSetReadPolicy",
        )

    def workbench_featureset_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench FeatureSets (FULL)"""
        policy_statements = [
            self.s3_full(),
            self.glue_catalog_full(),
            self.glue_databases_full(),
            self.athena_read(),
            self.featurestore_discovery(),
            self.featurestore_full(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_full(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchFeatureSetPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchFeatureSetPolicy",
        )

    def sagemaker_pass_role_policy(self) -> iam.PolicyStatement:
        """Create a policy statement allowing SageMaker to assume specific IAM roles.

        This policy enables SageMaker services to assume IAM roles on your behalf when:
        - Creating training jobs (assumes execution role for data access, logging, etc.)
        - Deploying endpoints (assumes role for model loading, inference logging)
        - Running batch transform jobs (assumes role for input/output data access)
        - Creating processing jobs (assumes role for data processing operations)

        Scoped to specific workbench roles:
        - Workbench-ExecutionRole: Main API execution role for Workbench tasks

        Returns:
            iam.PolicyStatement: Policy allowing SageMaker to assume specific Workbench roles
        """
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[
                f"arn:aws:iam::{self.account}:role/{self.execution_role_name}",
            ],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
        )

    def workbench_model_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.models_discovery(),
            self.models_read(),
            self.cloudwatch_logs(),
            self.cloudwatch_metrics(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchModelReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchModelReadPolicy",
        )

    def workbench_model_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.models_discovery(),
            self.models_full(),
            self.model_training(),
            self.model_training_logs(),
            self.ecr_policy_statement(),
            self.cloudwatch_logs(),
            self.cloudwatch_metrics(),
            self.sagemaker_pass_role_policy(),
            self.parameter_store_discover(),
            self.parameter_store_full(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchModelPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchModelPolicy",
        )

    def workbench_endpoint_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.models_discovery(),  # Endpoints get information from their internal Model(s)
            self.models_read(),
            self.endpoint_discover(),
            self.endpoint_read(),
            self.endpoint_monitoring_discovery(),
            self.endpoint_monitoring_schedules(),
            self.endpoint_monitoring_processing(),
            self.cloudwatch_logs(),
            self.cloudwatch_metrics(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchEndpointReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchEndpointReadPolicy",
        )

    def workbench_endpoint_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Models"""
        policy_statements = [
            self.models_discovery(),  # Endpoints get information from their internal Model(s)
            self.models_read(),
            self.endpoint_discover(),
            self.endpoint_full(),
            self.endpoint_data_quality(),
            self.endpoint_monitoring_discovery(),
            self.endpoint_monitoring_schedules(),
            self.endpoint_monitoring_processing(),
            self.cloudwatch_logs(),
            self.cloudwatch_metrics(),
            self.parameter_store_discover(),
            self.parameter_store_full(),
        ]
        return iam.ManagedPolicy(
            self,
            id="WorkbenchEndpointPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchEndpointPolicy",
        )

    def workbench_dataframe_store_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench DataFrame Store (READ-ONLY)"""
        policy_statements = [
            self.s3_read(),
            self.glue_jobs_s3_read(),
            self.glue_job_logs(),
            self.cloudwatch_logs(),
            self.parameter_store_read(),  # Get Workbench Bucket from Parameter Store
            self.dataframe_store_read(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchDFStoreReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchDFStoreReadPolicy",
        )

    def workbench_dataframe_store_full_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench DataFrame Store (FULL)"""
        policy_statements = [
            self.s3_read(),
            self.glue_jobs_s3_read(),
            self.glue_job_logs(),
            self.cloudwatch_logs(),
            self.parameter_store_read(),  # Get Workbench Bucket from Parameter Store
            self.dataframe_store_full(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchDFStoreFullPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchDFStoreFullPolicy",
        )

    def workbench_parameter_store_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Parameter Store (READ-ONLY)"""
        policy_statements = [
            self.s3_read(),
            self.glue_jobs_s3_read(),
            self.glue_job_logs(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_read(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchParameterStoreReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchParameterStoreReadPolicy",
        )

    def workbench_parameter_store_full_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Parameter Store (FULL)"""
        policy_statements = [
            self.s3_read(),
            self.glue_jobs_s3_read(),
            self.glue_job_logs(),
            self.cloudwatch_logs(),
            self.parameter_store_discover(),
            self.parameter_store_full(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchParameterStoreFullPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchParameterStoreFullPolicy",
        )

    def workbench_inference_store_read_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Inference Store (READ-ONLY)"""
        policy_statements = [
            self.s3_read(),
            self.glue_jobs_s3_read(),
            self.athena_query_results_s3(),
            self.glue_job_logs(),
            self.cloudwatch_logs(),
            self.glue_catalog_read(),
            self.glue_database_read_just_inference_store(),
            self.athena_read(),
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchInferenceStoreReadPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchInferenceStoreReadPolicy",
        )

    def workbench_inference_store_full_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the Workbench Inference Store (FULL)"""
        policy_statements = [
            self.s3_read(),
            self.glue_jobs_s3_read(),
            self.s3_full_just_inference_store(),
            self.athena_query_results_s3(),
            self.glue_job_logs(),
            self.cloudwatch_logs(),
            self.glue_catalog_read(),
            self.glue_database_full_just_inference_store(),
            self.athena_read(),
            self.parameter_store_read(),  # Get Workbench Bucket from Parameter Store
        ]

        return iam.ManagedPolicy(
            self,
            id="WorkbenchInferenceStoreFullPolicy",
            statements=policy_statements,
            managed_policy_name="WorkbenchInferenceStoreFullPolicy",
        )

    def create_execution_role(self) -> iam.Role:
        """Create the Workbench Execution Role for API-related tasks"""
        # Define the base assumed by principals with service principals
        base_principals = iam.CompositePrincipal(
            iam.ServicePrincipal("ecs-tasks.amazonaws.com"), iam.ServicePrincipal("sagemaker.amazonaws.com")
        )

        # Add SSO configuration to the principals
        assumed_by = self._create_sso_principals(base_principals)

        # Create the role with the trust relationships
        api_execution_role = iam.Role(
            self,
            id=self.execution_role_name,
            assumed_by=assumed_by,
            role_name=self.execution_role_name,
        )

        # Create and attach the Workbench managed policies to the role
        api_execution_role.add_to_policy(self.glue_jobs_discover())
        api_execution_role.add_to_policy(self.glue_jobs_full())
        api_execution_role.add_to_policy(self.glue_pass_role())
        api_execution_role.add_to_policy(self.batch_jobs_discover())
        api_execution_role.add_to_policy(self.batch_jobs_full())
        api_execution_role.add_to_policy(self.batch_pass_role())
        api_execution_role.add_to_policy(self.sqs_discover())
        api_execution_role.add_to_policy(self.sqs_full())
        api_execution_role.add_to_policy(self.parameter_store_discover())
        api_execution_role.add_to_policy(self.parameter_store_full())
        api_execution_role.add_to_policy(self.cloudwatch_logs())
        api_execution_role.add_to_policy(self.cloudwatch_monitor())
        api_execution_role.add_managed_policy(self.datasource_policy)
        api_execution_role.add_managed_policy(self.featureset_policy)
        api_execution_role.add_managed_policy(self.model_policy)
        api_execution_role.add_managed_policy(self.endpoint_policy)
        return api_execution_role

    def create_readonly_role(self) -> iam.Role:
        """Create the Workbench Read-Only Role for viewing resources"""

        # ECS for Dashboard (we might switch to read-only later)
        base_principals = iam.CompositePrincipal(iam.ServicePrincipal("ecs-tasks.amazonaws.com"))

        # Add SSO configuration to the principals
        assumed_by = self._create_sso_principals(base_principals)
        readonly_role = iam.Role(
            self,
            id=self.readonly_role_name,
            assumed_by=assumed_by,
            role_name=self.readonly_role_name,
        )

        # Add our read-only policies here
        readonly_role.add_to_policy(self.glue_jobs_discover())
        readonly_role.add_to_policy(self.glue_jobs_read())
        readonly_role.add_to_policy(self.parameter_store_discover())
        readonly_role.add_to_policy(self.parameter_store_read())
        readonly_role.add_to_policy(self.cloudwatch_logs())
        readonly_role.add_managed_policy(self.datasource_read_policy)
        readonly_role.add_managed_policy(self.featureset_read_policy)
        readonly_role.add_managed_policy(self.model_read_policy)
        readonly_role.add_managed_policy(self.endpoint_read_policy)
        return readonly_role

    def _create_sso_instructions(self, execution_role: iam.Role, readonly_role: iam.Role):
        """Print SSO setup instructions to console"""
        if self.sso_groups:
            # Build ARNs manually since tokens aren't resolved yet
            execution_arn = f"arn:aws:iam::{self.account}:role/{self.execution_role_name}"
            readonly_arn = f"arn:aws:iam::{self.account}:role/{self.readonly_role_name}"

            print("\n" + "=" * 60)
            print("🔧  SSO SETUP REQUIRED (If not already done)  🔧")
            print("=" * 60)
            print("Have your SSO Administrator add these roles to your permission set:")
            print(f"  • {execution_arn}")
            print(f"  • {readonly_arn}")
            print()
            print("For multi-account deployments, you can add both lines for each account:")
            print("  • arn:aws:iam::<account_id>:role/Workbench-ExecutionRole")
            print("  • arn:aws:iam::<account_id>:role/Workbench-ReadOnlyRole")
            print("=" * 60 + "\n")

    def create_lambda_role(self) -> iam.Role:
        """Create the Workbench Lambda Role."""
        lambda_assumed_by = iam.ServicePrincipal("lambda.amazonaws.com")
        lambda_role = iam.Role(
            self,
            id=self.lambda_role_name,
            assumed_by=lambda_assumed_by,
            role_name=self.lambda_role_name,
        )
        # Add AWS managed policy for Lambda basic execution
        lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )

        # Add a subset of policies for the Lambda Role
        lambda_role.add_to_policy(self.parameter_store_full())
        lambda_role.add_to_policy(self.cloudwatch_logs())
        lambda_role.add_to_policy(self.batch_jobs_discover())
        lambda_role.add_to_policy(self.batch_jobs_full())
        lambda_role.add_to_policy(self.batch_pass_role())
        return lambda_role

    def create_glue_role(self) -> iam.Role:
        """Create the Workbench Glue Role."""
        glue_assumed_by = iam.ServicePrincipal("glue.amazonaws.com")
        glue_role = iam.Role(
            self,
            id=self.glue_role_name,
            assumed_by=glue_assumed_by,
            role_name=self.glue_role_name,
        )

        # Add a subset of policies for the Glue Role
        glue_role.add_to_policy(self.glue_job_logs())
        glue_role.add_to_policy(self.glue_connections())
        glue_role.add_to_policy(self.cloudwatch_logs())
        glue_role.add_to_policy(self.vpc_discovery())
        glue_role.add_to_policy(self.vpc_network_interface_management())
        glue_role.add_to_policy(self.parameter_store_full())
        glue_role.add_managed_policy(self.datasource_policy)
        glue_role.add_managed_policy(self.featureset_policy)
        glue_role.add_managed_policy(self.model_policy)
        glue_role.add_managed_policy(self.endpoint_policy)
        return glue_role

    def create_batch_role(self) -> iam.Role:
        """Create the Workbench Batch Role."""
        batch_assumed_by = iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        batch_role = iam.Role(
            self,
            id=self.batch_role_name,
            assumed_by=batch_assumed_by,
            role_name=self.batch_role_name,
        )

        # Add policies for the Batch Role
        batch_role.add_to_policy(self.batch_job_logs())
        batch_role.add_to_policy(self.cloudwatch_logs())
        batch_role.add_to_policy(self.parameter_store_full())
        batch_role.add_to_policy(self.dataframe_store_full())
        batch_role.add_managed_policy(self.datasource_policy)
        batch_role.add_managed_policy(self.featureset_policy)
        batch_role.add_managed_policy(self.model_policy)
        batch_role.add_managed_policy(self.endpoint_policy)
        return batch_role

    def _create_sso_principals(self, base_principals: iam.CompositePrincipal = None) -> iam.CompositePrincipal:
        """
        Create principals with SSO trust relationship configuration.

        Args:
            base_principals: Optional existing CompositePrincipal to extend

        Returns:
            CompositePrincipal with SSO configuration applied
        """
        assumed_by = base_principals or iam.CompositePrincipal()

        # If sso_groups are provided, configure trust relationship for AWS SSO integration
        # AWS SSO creates roles with two different ARN patterns depending on configuration:
        if self.sso_groups:
            # Build lists to hold all ARN patterns for all groups
            sso_group_arns_pattern_1 = []
            sso_group_arns_pattern_2 = []

            # Generate ARN patterns for each SSO group
            for sso_group in self.sso_groups:
                # Pattern 1: Direct group-based SSO role (no permission set in path)
                # Used when SSO groups are mapped directly to roles
                sso_group_arn_1 = (
                    f"arn:aws:iam::{self.account}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_{sso_group}_*"
                )
                sso_group_arns_pattern_1.append(sso_group_arn_1)

                # Pattern 2: Permission set-based SSO role (includes permission set ID in path)
                # Used when SSO groups are assigned via permission sets (more common in enterprise setups)
                # The middle wildcard (*) represents the permission set identifier
                sso_group_arn_2 = (
                    f"arn:aws:iam::{self.account}:role/aws-reserved/sso.amazonaws.com/*/AWSReservedSSO_{sso_group}_*"
                )
                sso_group_arns_pattern_2.append(sso_group_arn_2)

            # Combine both patterns into a single list for the condition
            # Both patterns are required for maximum compatibility across different SSO configurations
            # This is particularly important for Lake Formation and enterprise SSO integrations
            all_sso_arns = sso_group_arns_pattern_1 + sso_group_arns_pattern_2

            condition = {"ArnLike": {"aws:PrincipalArn": all_sso_arns}}
            assumed_by.add_principals(iam.AccountPrincipal(self.account).with_conditions(condition))
        else:
            # Fallback: Allow any principal in the account to assume this role
            assumed_by.add_principals(iam.AccountPrincipal(self.account))

        return assumed_by
