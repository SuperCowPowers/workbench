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
                self.log.error('aws glue create-database --database-input \'{"Name": "workbench"}\'')
                sys.exit(1)
            else:
                self.log.error(f"Unexpected error: {e}")
                sys.exit(1)

    def check_s3_bucket_subfolders(self):
        """Check if the Workbench S3 Bucket is set up and has the correct sub-folders"""

        self.log.info("*** AWS Workbench Bucket Check ***")
        s3 = self.aws_clamp.boto3_session.resource("s3")
        bucket = s3.Bucket(self.workbench_bucket)

        # Check if the bucket exists
        if bucket.creation_date is None:
            self.log.critical(f"The {self.workbench_bucket} bucket does not exist")
            sys.exit(1)
        else:
            self.log.info(f"The {self.workbench_bucket} bucket exists")

        # Check if the sub-folders exists
        sub_folders = ["incoming-data", "data-sources", "feature-sets", "athena-queries"]

        # Get all object prefixes in the bucket
        prefixes = set()
        for obj in bucket.objects.all():
            prefix = obj.key.split("/")[0]
            prefixes.add(prefix)

        # Check for the existence of the sub-folders
        for folder in sub_folders:
            if folder in prefixes:
                self.log.info(f"The {folder} prefix exists")
            else:
                self.log.info(f"The {folder} prefix does not exist...which is fine...")

    def check(self):
        """Check if the AWS Account is set up Correctly"""
        self.log.info("*** Caller/Base AWS Identity Check ***")
        self.aws_clamp.check_aws_identity()
        print("\n")

        self.log.info("*** AWS Assumed Role Check ***")
        self.aws_clamp.check_assumed_role()
        print("\n")

        self.log.info("*** AWS S3 Access Check ***")
        self.aws_clamp.check_s3_access()
        print("\n")

        # Check that the Workbench S3 Bucket and Sub-folders are created
        self.check_s3_bucket_subfolders()

        self.log.info("*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.aws_clamp.sagemaker_client()
        for feature_group in sm_client.list_feature_groups()["FeatureGroupSummaries"]:
            self.log.info(str(feature_group))
        print("\n")

        # Check that the Glue Database exist
        self.log.info("*** AWS Glue Database Check ***")
        for catalog_db in ["workbench", "sagemaker_featurestore"]:
            self.ensure_aws_catalog_db(catalog_db)
        print("\n")

        self.log.info("AWS Account Clamp: AOK!")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    account_check = AWSAccountCheck()

    # Check that out AWS Account Clamp is working AOK
    account_check.check()
