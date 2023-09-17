from aws_cdk import Stack, aws_iam as iam, aws_s3 as s3, aws_glue as glue
from constructs import Construct


class SageworksStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        s3_bucket_name,
        sageworks_role_name,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create the SageWorks Artifact Bucket (must be created before Roles)
        self.artifact_bucket = self.add_artifact_bucket(s3_bucket_name)

        # Create our main SageWorks Execution Role and the Glue Service Role
        self.sageworks_execution_role = self.create_execution_role(sageworks_role_name, iam.AnyPrincipal())
        self.glue_service_role = self.create_execution_role(
            "AWSGlueServiceRole-Sageworks", iam.ServicePrincipal("glue.amazonaws.com")
        )

        # Create the SageWorks Data Catalog Databases
        self.create_data_catalog_databases()

    def add_artifact_bucket(self, bucket_name) -> s3.Bucket:
        return s3.Bucket(
            self,
            id=bucket_name,
            bucket_name=bucket_name,
        )

    def create_execution_role(self, role_name, assumed_by) -> iam.Role:
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
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:*", "s3-object-lambda:*"],
                resources=[
                    "arn:aws:s3:::aws-athena-query-results*/*",
                    f"arn:aws:s3:::{self.artifact_bucket.bucket_name}/*",
                ],
            )
        )
        return execution_role

    def create_data_catalog_databases(self) -> None:
        # Create two Data Catalog Databases using CfnDatabase (sageworks and sagemaker_featurestore)
        self.add_data_catalog_database("sageworks")
        self.add_data_catalog_database("sagemaker_featurestore")

    def add_data_catalog_database(self, database_name: str) -> None:
        glue.CfnDatabase(
            self, f"{database_name}_database", catalog_id=self.account, database_input={"name": database_name}
        )
