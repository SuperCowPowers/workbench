"""AWSAccountClamp provides logic/functionality over a set of AWS IAM Services"""

import boto3
from botocore.exceptions import (
    ClientError,
    UnauthorizedSSOTokenError,
    TokenRetrievalError,
)
from botocore.client import BaseClient
import logging
from sagemaker.session import Session as SageSession

# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_session import AWSSession
from sageworks.utils.config_manager import ConfigManager, FatalConfigError


class AWSAccountClamp:
    """AWSAccountClamp: Singleton class for connecting to an AWS Account"""

    _instance = None  # Class attribute to hold the singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AWSAccountClamp, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """AWSAccountClamp Initialization"""
        if hasattr(self, "_initialized") and self._initialized:
            return  # Prevent reinitialization

        # Setup code (only runs once)
        self.log = logging.getLogger("sageworks")
        self.log.info("Creating the AWSAccountClamp Singleton...")

        # ConfigManager and AWS setup
        self.cm = ConfigManager()
        if not self.cm.config_okay():
            self.log.error("SageWorks Configuration Incomplete...")
            self.log.error("Run the 'sageworks' command and follow the prompts...")
            raise FatalConfigError()
        self.sageworks_bucket_name = self.cm.get_config("SAGEWORKS_BUCKET")
        self.aws_session = AWSSession()
        self.boto3_session = self.aws_session.boto3_session

        # Check our AWS Identity
        try:
            self.account_id = boto3.client("sts").get_caller_identity()["Account"]
            self.region = boto3.session.Session().region_name
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError):
            self.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            raise FatalConfigError()

        # Check our SageWorks API Key and Load the License
        self.log.info("Checking SageWorks API License...")
        self.cm.load_and_check_license(self.account_id)
        self.cm.print_license_info()

        # Mark the instance as initialized
        self._initialized = True

    def check_aws_identity(self) -> bool:
        """Check the AWS Identity currently active"""
        # Check AWS Identity Token
        sts = boto3.client("sts")
        try:
            identity = sts.get_caller_identity()
            self.log.info("AWS Account Info:")
            self.log.info(f"Account: {identity['Account']}")
            self.log.info(f"Identity ARN: {identity['Arn']}")
            self.log.info(f"Region: {self.region}")
            return True
        except (ClientError, UnauthorizedSSOTokenError):
            msg = "AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            self.log.critical(msg)
            raise RuntimeError(msg)

    def get_aws_account_info(self) -> dict:
        """Get the AWS Account Information

        Returns:
            dict: A dictionary of the AWS Account Information
        """
        info = {}
        sts = boto3.client("sts")
        try:
            identity = sts.get_caller_identity()
            info["Account"] = identity["Account"]
            info["IdentityArn"] = identity["Arn"]
            info["Region"] = self.region
            return info
        except (ClientError, UnauthorizedSSOTokenError):
            self.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            return info

    def check_s3_access(self) -> bool:
        s3 = self.boto3_session.client("s3")
        results = s3.list_buckets()
        for bucket in results["Buckets"]:
            self.log.info(f"\t{bucket['Name']}")
        return True

    def sagemaker_session(self) -> "SageSession":
        """Create a sageworks SageMaker session (using our boto3 refreshable session)

        Returns:
            SageSession: A SageMaker session object
        """
        return SageSession(boto_session=self.boto3_session)

    def sagemaker_client(self) -> BaseClient:
        """Create a sageworks SageMaker client (using our boto3 refreshable session)

        Returns:
            BaseClient: A SageMaker client object
        """
        return self.boto3_session.client("sagemaker")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    aws_account_clamp = AWSAccountClamp()

    # Check out that AWS Account Clamp is working AOK
    """Check if the AWS Account is Setup Correctly"""
    print("*** AWS Identity Check ***")
    aws_account_clamp.check_aws_identity()
    print("Identity Check Success...")

    print("*** AWS S3 Access Check ***")
    aws_account_clamp.check_s3_access()
    print("S3 Access Check Success...")

    print("*** AWS Sagemaker Session/Client Check ***")
    sm_client = aws_account_clamp.sagemaker_client()
    print(sm_client.list_feature_groups()["FeatureGroupSummaries"])
