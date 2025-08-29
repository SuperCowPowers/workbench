"""AWSAccountCheck runs a bunch of tests/checks to ensure Workbench AWS Setup"""

import sys
import logging
import awswrangler as wr
from botocore.exceptions import ClientError

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager, FatalConfigError


class AWSAccountCheck:
    """AWSAccountCheck runs a bunch of tests/checks to ensure Workbench AWS Setup"""

    def __init__(self):
        """AWSAccountCheck Initialization"""
        self.log = logging.getLogger("workbench")

        # Create the AWSAccountClamp Class
        self.aws_clamp = AWSAccountClamp()

        # Grab our Workbench Bucket
        cm = ConfigManager()
        if not cm.config_okay():
            self.log.error("Workbench Configuration Incomplete...")
            self.log.error("Run the 'workbench' command and follow the prompts...")
            raise FatalConfigError()

        self.workbench_bucket = cm.get_config("WORKBENCH_BUCKET")

    def ensure_aws_catalog_db(self, catalog_db: str):
        """Ensure that the AWS Data Catalog Database exists"""
        self.log.important(f"Ensuring that the AWS Data Catalog Database {catalog_db} exists...")
        try:
            wr.catalog.create_database(catalog_db, exist_ok=True, boto3_session=self.aws_clamp.boto3_session)
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                self.log.error(f"AccessDeniedException {e}")
                self.log.error(f"Access denied while trying to create/access the catalog database '{catalog_db}'.")
                self.log.error("Create the database manually in the AWS Glue Console, or run this command:")
                self.log.error(f'aws glue create-database --database-input \'{{"Name": "{catalog_db}"}}\'')
                sys.exit(1)
            else:
                self.log.error(f"Unexpected error: {e}")
                sys.exit(1)

    def ensure_athena_results_bucket(self):
        """Ensure that the Athena results S3 bucket exists"""
        bucket_name = f"aws-athena-query-results-{self.aws_clamp.account_id}-{self.aws_clamp.region}"
        self.log.important(f"Checking if Athena results bucket {bucket_name} exists...")

        try:
            s3_client = self.aws_clamp.boto3_session.client("s3")
            s3_client.head_bucket(Bucket=bucket_name)
            self.log.info(f"Athena results bucket {bucket_name} already exists")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.log.error(f"Athena results bucket {bucket_name} does not exist.")
                self.log.error("Create the bucket manually using this AWS CLI command:")
                self.log.error(f"aws s3 mb s3://{bucket_name}")
                sys.exit(1)
            else:
                self.log.error(f"Error checking bucket {bucket_name}: {e}")
                sys.exit(1)

    def check(self):
        """Check if the AWS Account is set up Correctly"""
        self.log.info("*** Caller/Base AWS Identity Check ***")
        self.aws_clamp.check_aws_identity()
        print("\n")

        self.log.info("*** AWS Assumed Role Check ***")
        self.aws_clamp.check_assumed_role()
        print("\n")

        self.log.info("*** AWS Workbench Bucket Check ***")
        self.aws_clamp.check_workbench_bucket()
        print("\n")

        self.log.info("*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.aws_clamp.sagemaker_client()
        for feature_group in sm_client.list_feature_groups()["FeatureGroupSummaries"]:
            self.log.info(str(feature_group))
        print("\n")

        # Check that the Glue Databases exist
        self.log.info("*** AWS Glue Databases Check ***")
        for catalog_db in ["workbench", "sagemaker_featurestore", "inference_store"]:
            self.ensure_aws_catalog_db(catalog_db)
        print("\n")

        # Check that the Athena Results Bucket exists
        self.log.info("*** AWS Athena Results Bucket Check ***")
        self.ensure_athena_results_bucket()
        print("\n")

        self.log.info("AWS Account Clamp: AOK!")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    account_check = AWSAccountCheck()

    # Check that out AWS Account Clamp is working AOK
    account_check.check()
