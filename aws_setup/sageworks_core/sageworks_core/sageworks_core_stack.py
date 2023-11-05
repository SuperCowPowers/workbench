from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_glue as glue,
)
from constructs import Construct
from typing import Any
from dataclasses import dataclass


@dataclass
class SageworksCoreStackProps:
    sageworks_bucket: str
    sageworks_role_name: str
    sso_role_arn: str


class SageworksCoreStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        props: SageworksCoreStackProps,
        **kwargs: Any,
    ) -> None:
        desc = "SageWorks Core Stack: Execution Role and Artifact Bucket"
        super().__init__(scope, construct_id, description=desc, **kwargs)

        # Grab our properties
        self.account_id = env.account
        self.sageworks_bucket = props.sageworks_bucket
        self.sageworks_role_name = props.sageworks_role_name
        self.sso_role_arn = props.sso_role_arn

        # Create the SageWorks Artifact Bucket (must be created before Roles)
        self.artifact_bucket = self.add_artifact_bucket(self.sageworks_bucket)

        # Create our main SageWorks Execution Role
        self.sageworks_execution_role = self.create_execution_role(self.sageworks_role_name, iam.AccountPrincipal(self.account_id))

        # Create a SageWorks Execution Role for AWS Glue
        # Note: This is a duplicate, but Glue Jobs require a specific role name and this role can be assumed by Glue Services
        self.glue_service_role = self.create_execution_role(
            "AWSGlueServiceRole-Sageworks", iam.ServicePrincipal("glue.amazonaws.com")
        )

    def add_artifact_bucket(self, bucket_name) -> s3.Bucket:
        return s3.Bucket(
            self,
            id=bucket_name,
            bucket_name=bucket_name,
        )

    def create_execution_role(self, role_name, assumed_by) -> iam.Role:
        # Create the SageWorks Execution Role with the following policies:
        #   - AmazonSageMakerFullAccess
        #   - service-role/AWSGlueServiceRole
        execution_role = iam.Role(
            self,
            id=role_name,
            assumed_by=assumed_by,
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
            role_name=role_name,
        )

        # First policy statement for S3 and S3 Object Lambda
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:*", "s3-object-lambda:*"],
                resources=[
                    "arn:aws:s3:::aws-athena-query-results*/*",
                    f"arn:aws:s3:::{self.artifact_bucket.bucket_name}/*",
                ],
            )
        )

        # Second policy statement for ECS DescribeServices
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:DescribeServices"],
                resources=["*"],
            )
        )

        # Third policy statement for ELB DescribeLoadBalancers
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["elasticloadbalancing:DescribeLoadBalancers"],
                resources=["*"],
            )
        )
        return execution_role
