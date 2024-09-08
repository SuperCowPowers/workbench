"""AWSAccountCheck runs a bunch of tests/checks to ensure SageWorks AWS Setup"""

import sys
import logging
import awswrangler as wr

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager, FatalConfigError


class AWSAccountCheck:
    """AWSAccountCheck runs a bunch of tests/checks to ensure SageWorks AWS Setup"""

    def __init__(self):
        """AWSAccountCheck Initialization"""
        self.log = logging.getLogger("sageworks")

        # Create the AWSAccountClamp Class
        self.aws_clamp = AWSAccountClamp()

        # Grab our SageWorks Bucket
        cm = ConfigManager()
        if not cm.config_okay():
            self.log.error("SageWorks Configuration Incomplete...")
            self.log.error("Run the 'sageworks' command and follow the prompts...")
            raise FatalConfigError()

        self.sageworks_bucket = cm.get_config("SAGEWORKS_BUCKET")

    def ensure_aws_catalog_db(self, catalog_db: str):
        """Ensure that the AWS Data Catalog Database exists"""
        self.log.important(f"Ensuring that the AWS Data Catalog Database {catalog_db} exists...")
        wr.catalog.create_database(catalog_db, exist_ok=True, boto3_session=self.aws_clamp.boto3_session)

    def check_s3_bucket_subfolders(self):
        """Check if the SageWorks S3 Bucket is set up and has the correct sub-folders"""

        self.log.info("*** AWS SageWorks Bucket Check ***")
        s3 = self.aws_clamp.boto3_session.resource("s3")
        bucket = s3.Bucket(self.sageworks_bucket)

        # Check if the bucket exists
        if bucket.creation_date is None:
            self.log.critical(f"The {self.sageworks_bucket} bucket does not exist")
            sys.exit(1)
        else:
            self.log.info(f"The {self.sageworks_bucket} bucket exists")

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
        """Check if the AWS Account is Setup Correctly"""
        self.log.info("*** AWS Identity Check ***")
        self.aws_clamp.check_aws_identity()
        self.log.info("Identity Check Success...")

        self.log.info("*** AWS S3 Access Check ***")
        self.aws_clamp.check_s3_access()
        self.log.info("S3 Access Check Success...")

        # Check that the SageWorks S3 Bucket and Sub-folders are created
        self.check_s3_bucket_subfolders()

        self.log.info("*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.aws_clamp.sagemaker_client()
        for feature_group in sm_client.list_feature_groups()["FeatureGroupSummaries"]:
            self.log.info(str(feature_group))

        # Check that the Glue Database exist
        self.log.info("*** AWS Glue Database Check ***")
        for catalog_db in ["sageworks", "sagemaker_featurestore"]:
            self.ensure_aws_catalog_db(catalog_db)

        self.log.info("\n\nAWS Account Clamp: AOK!")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    aws_clamp = AWSAccountCheck()

    # Check that out AWS Account Clamp is working AOK
    aws_clamp.check()
