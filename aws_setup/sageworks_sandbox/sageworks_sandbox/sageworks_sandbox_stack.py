from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


class SageworksStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, company_name: str, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.sageworks_execution_role = self.create_execution_role(
            "SageWorksExecutionRole"
        )
        self.sageworks_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:*", "s3-object-lambda:*"],
                resources=[
                    "arn:aws:s3:::aws-athena-query-results",
                    "arn:aws:s3:::*sageworks-artifacts*",
                ],
            )
        )
        self.add_artifact_bucket(company_name)

        # TODO: Use this if we need a duplicate role
        # self.glue_service_role = self.create_execution_role("AWSGlueServiceRole-Sageworks")

    def create_execution_role(self, role_name) -> iam.Role:
        return iam.Role(
            self,
            id=role_name,
            assumed_by=iam.AnyPrincipal(),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSGlueServiceRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

    def add_artifact_bucket(self, company_name) -> None:
        self.artifact_bucket = s3.Bucket(
            self,
            f"com.{company_name}-sageworks-artifacts",
        )
