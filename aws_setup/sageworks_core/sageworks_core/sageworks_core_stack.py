from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
)
from constructs import Construct
from typing import Any, List
from dataclasses import dataclass, field


@dataclass
class SageworksCoreStackProps:
    sageworks_bucket: str
    sageworks_role_name: str
    sso_group: str
    additional_buckets: List[str] = field(default_factory=list)


class SageworksCoreStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        props: SageworksCoreStackProps,
        **kwargs: Any,
    ) -> None:
        desc = "SageWorks Core Stack: SageWorks-ExecutionRole (API) and  AWSGlueServiceRole-SageWorks (Glue)"
        super().__init__(scope, construct_id, description=desc, **kwargs)

        # Grab our properties
        self.account_id = env.account
        self.sageworks_bucket = props.sageworks_bucket
        self.sageworks_role_name = props.sageworks_role_name
        self.sso_group = props.sso_group
        self.additional_buckets = props.additional_buckets

        # Create a list of buckets
        athena_bucket = "aws-athena-query-results*"
        sagemaker_bucket = "sagemaker-{region}-{account_id}*"
        self.bucket_list = [self.sageworks_bucket, athena_bucket, sagemaker_bucket] + self.additional_buckets
        self.bucket_arns = self._bucket_names_to_arns(self.bucket_list)

        # Create our main SageWorks Execution Role
        self.sageworks_api_execution_role = self.create_api_execution_role()

    def _bucket_names_to_arns(self, bucket_list: list[str]) -> list[str]:
        """Convert a list of dynamic bucket names to ARNs."""
        arns = []
        for bucket_name_template in bucket_list:
            # Dynamically construct the bucket name
            bucket_name = bucket_name_template.format(region=self.region, account_id=self.account_id)
            bucket_arn = f"arn:aws:s3:::{bucket_name}"
            arns.append(bucket_arn)
            arns.append(f"{bucket_arn}/*")
        return arns

    @staticmethod
    def s3_list_policy_statement() -> iam.PolicyStatement:
        """Create policy statement for listing S3 buckets

        Returns:
           iam.PolicyStatement: A policy statements for listing S3 buckets
        """
        list_all_buckets_policy = iam.PolicyStatement(
            actions=["s3:ListAllMyBuckets"],
            resources=["*"],  # ListAllMyBuckets applies to all buckets
        )
        return list_all_buckets_policy

    def s3_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for S3 access.

        Returns:
            iam.PolicyStatement: The policy statement for S3 access.
        """
        return iam.PolicyStatement(
            actions=[
                # Define the S3 actions you need
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetBucketAcl",
            ],
            resources=self.bucket_arns,
        )

    @staticmethod
    def s3_public_policy_statement() -> iam.PolicyStatement:
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

    @staticmethod
    def glue_job_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for AWS Glue job-related actions.

        Returns:
            iam.PolicyStatement: The policy statement for AWS Glue job operations.
        """
        return iam.PolicyStatement(
            actions=[
                "glue:GetJobs",  # Retrieve metadata about all the Glue jobs
                "glue:GetJobRuns",  # Retrieve metadata about Glue job runs
                "glue:GetJob",  # Retrieve metadata about a specific Glue job
                "glue:StartJobRun",  # Example additional job-related action
                "glue:StopJobRun",  # Another example job-related action
            ],
            resources=["*"],  # Broad permission necessary for job management
        )

    def glue_catalog_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for permissions related to the Glue Data Catalog.

        Returns:
            iam.PolicyStatement: The policy statement for Glue Data Catalog access.
        """
        catalog_arn = f"arn:aws:glue:{self.region}:{self.account}:catalog"

        return iam.PolicyStatement(
            actions=[
                # Permissions for catalog/database-level actions
                "glue:CreateDatabase",
                "glue:GetDatabase",
                "glue:GetDatabases",
                "glue:SearchTables",
                "glue:GetTable",
                "glue:GetTables",
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
                "glue:GetPartition",
                "glue:GetPartitions",
            ],
            resources=[catalog_arn],
        )

    def glue_database_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for AWS Glue Data Catalog read and write access, limited to specific databases.

        Returns:
            iam.PolicyStatement: The policy statement for AWS Glue Data Catalog access.
        """
        # Construct ARNs for the specific databases
        sageworks_database_arn = f"arn:aws:glue:{self.region}:{self.account}:database/sageworks"
        sageworks_table_arn = f"arn:aws:glue:{self.region}:{self.account}:table/sageworks/*"
        sagemaker_featurestore_database_arn = (
            f"arn:aws:glue:{self.region}:{self.account}:database/sagemaker_featurestore"
        )
        sagemaker_table_arn = f"arn:aws:glue:{self.region}:{self.account}:table/sagemaker_featurestore/*"

        return iam.PolicyStatement(
            actions=[
                # Permissions for database-level actions, we need to 'get' the database
                "glue:GetDatabase",
                # Permissions for table-level actions, focusing on read/write operations
                "glue:GetTable",
                "glue:GetTables",
                "glue:UpdateTable",
                "glue:CreateTable",
                "glue:DeleteTable",
                # Actions for working with data in tables
                "glue:GetPartition",
                "glue:GetPartitions",
            ],
            resources=[
                sageworks_database_arn,
                f"{sageworks_database_arn}/*",
                sageworks_table_arn,
                sagemaker_featurestore_database_arn,
                f"{sagemaker_featurestore_database_arn}/*",
                sagemaker_table_arn,
            ],
        )

    def athena_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for Athena actions that involve S3 buckets.

        Returns:
            iam.PolicyStatement: The policy statement for Athena S3 bucket access.
        """
        return iam.PolicyStatement(
            actions=[
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
                "sagemaker:ListModelPackages",
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

    @staticmethod
    def endpoint_list_monitoring_policy_statement() -> iam.PolicyStatement:
        """Create a policy statement for listing all SageMaker monitoring schedules.

        Returns:
            iam.PolicyStatement: The policy statement for listing SageMaker monitoring schedules.
        """
        return iam.PolicyStatement(
            actions=[
                "sagemaker:ListMonitoringSchedules",
            ],
            resources=["*"],  # ListMonitoringSchedules does not support specific resources
        )

    def pipeline_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for running SageMaker Pipelines.

        Returns:
            iam.PolicyStatement: The policy statement for running SageMaker Pipelines.
        """

        # Sagemaker Pipeline Processing Jobs ARN
        processing_resources = f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*"

        return iam.PolicyStatement(
            actions=[
                # Actions for Jobs
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob",
                "sagemaker:ListProcessingJobs",
                "sagemaker:StopProcessingJob",
                # Additional actions
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:DeleteTags",
            ],
            resources=[processing_resources],
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
        """Create a policy statement for additional permissions needed by SageWorks Dashboard.

        Returns:
            iam.PolicyStatement: The policy statement needed by SageWorks Dashboard.
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

    def sageworks_datasource_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the SageWorks DataSources"""
        policy_statements = [
            self.s3_list_policy_statement(),
            self.s3_policy_statement(),
            self.s3_public_policy_statement(),
            self.glue_job_policy_statement(),
            self.glue_catalog_policy_statement(),
            self.glue_database_policy_statement(),
            self.athena_policy_statement(),
            self.athena_workgroup_policy_statement(),
            self.parameter_store_policy_statement(),
        ]

        return iam.ManagedPolicy(
            self,
            id="SageWorksDataSourcePolicy",
            statements=policy_statements,
            managed_policy_name="SageWorksDataSourcePolicy",
        )

    def sageworks_featureset_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the SageWorks FeatureSets"""
        policy_statements = [
            self.s3_policy_statement(),
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
            id="SageWorksFeatureSetPolicy",
            statements=policy_statements,
            managed_policy_name="SageWorksFeatureSetPolicy",
        )

    """In SageMaker, certain operations, such as creating training jobs, endpoint deployments, or batch transform jobs,
       require SageMaker to assume an IAM role. This role provides SageMaker with permissions to access AWS resources 
       on your behalf, such as reading training data from S3, writing model artifacts, or logging to CloudWatch."""

    def sagemaker_pass_role_policy_statement(self) -> iam.PolicyStatement:
        """Create a policy statement for SageMaker to assume the Execution Role

        Args:
            sageworks_api_role (iam.Role): The SageWorks Execution Role
        """
        return iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=["arn:aws:iam::*:role/*"],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
        )

    def sageworks_model_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the SageWorks Models"""
        policy_statements = [
            self.model_list_policy_statement(),
            self.model_policy_statement(),
            self.model_training_statement(),
            self.model_training_log_statement(),
            self.pipeline_policy_statement(),
            self.ecr_policy_statement(),
            self.cloudwatch_policy_statement(),
            self.sagemaker_pass_role_policy_statement(),
            self.parameter_store_policy_statement(),
        ]
        return iam.ManagedPolicy(
            self,
            id="SageWorksModelPolicy",
            statements=policy_statements,
            managed_policy_name="SageWorksModelPolicy",
        )

    def sageworks_endpoint_policy(self) -> iam.ManagedPolicy:
        """Create a managed policy for the SageWorks Models"""
        policy_statements = [
            self.endpoint_list_policy_statement(),
            self.endpoint_policy_statement(),
            self.endpoint_list_monitoring_policy_statement(),
            self.cloudwatch_policy_statement(),
            self.parameter_store_policy_statement(),
        ]
        return iam.ManagedPolicy(
            self,
            id="SageWorksEndpointPolicy",
            statements=policy_statements,
            managed_policy_name="SageWorksEndpointPolicy",
        )

    def create_api_execution_role(self) -> iam.Role:
        """Create the SageWorks Execution Role for API-related tasks"""

        # Define the base assumed by principals with ECS service principal
        assumed_by = iam.CompositePrincipal(
            iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            iam.ServicePrincipal("sagemaker.amazonaws.com"),
            iam.ServicePrincipal("glue.amazonaws.com"),
        )

        # If sso_group is provided, add the condition to the trust relationship
        if self.sso_group:
            sso_group_arn_1 = (
                f"arn:aws:iam::{self.account_id}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_{self.sso_group}_*"
            )
            sso_group_arn_2 = f"arn:aws:iam::{self.account_id}:role/aws-reserved/sso.amazonaws.com/*/AWSReservedSSO_{self.sso_group}_*"
            condition = {"ArnLike": {"aws:PrincipalArn": [sso_group_arn_1, sso_group_arn_2]}}
            assumed_by.add_principals(iam.AccountPrincipal(self.account_id).with_conditions(condition))
        else:
            assumed_by.add_principals(iam.AccountPrincipal(self.account_id))

        # Create the role with the trust relationships
        api_execution_role = iam.Role(
            self,
            id=self.sageworks_role_name,
            assumed_by=assumed_by,
            role_name=self.sageworks_role_name,
        )

        # Create and attach the SageWorks managed policies to the role
        api_execution_role.add_managed_policy(self.sageworks_datasource_policy())
        api_execution_role.add_managed_policy(self.sageworks_featureset_policy())
        api_execution_role.add_managed_policy(self.sageworks_model_policy())
        api_execution_role.add_managed_policy(self.sageworks_endpoint_policy())

        return api_execution_role
