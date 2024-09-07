"""AWSAccountClamp provides logic/functionality over a set of AWS IAM Services"""

import boto3
from botocore.exceptions import (
    ClientError,
    UnauthorizedSSOTokenError,
    TokenRetrievalError,
)
from botocore.client import BaseClient
import logging

# We import SageSession lazily, so we'll leave this hint here for type checkers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagemaker.session import Session as SageSession

# SageWorks Imports
from sageworks.aws_service_broker.aws_session import AWSSession
from sageworks.utils.config_manager import ConfigManager, FatalConfigError


class AWSAccountClamp:

    # Initialize Class Attributes
    log = None
    cm = None
    sageworks_bucket_name = None
    account_id = None
    region = None
    boto3_session = None
    instance = None

    def __new__(cls):
        """AWSAccountClamp Singleton Pattern"""
        if cls.instance is None:
            # Initialize class attributes here
            cls.log = logging.getLogger("sageworks")
            cls.log.info("Creating the AWSAccountClamp Singleton...")

            # Pull in our config from the config manager
            cls.cm = ConfigManager()
            if not cls.cm.config_okay():
                cls.log.error("SageWorks Configuration Incomplete...")
                cls.log.error("Run the 'sageworks' command and follow the prompts...")
                raise FatalConfigError()
            cls.sageworks_bucket_name = cls.cm.get_config("SAGEWORKS_BUCKET")

            # Grab our AWS Boto3 Session
            cls.aws_session = AWSSession()
            cls.boto3_session = cls.aws_session.boto3_session

            # Check our AWS Identity
            try:
                cls.account_id = boto3.client("sts").get_caller_identity()["Account"]
                cls.region = boto3.session.Session().region_name
            except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError):
                cls.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
                raise FatalConfigError()

            # Check our SageWorks API Key and Load the License
            cls.log.info("Checking SageWorks API License...")
            cls.cm.load_and_check_license(cls.account_id)
            cls.cm.print_license_info()

            # Create the Singleton Instance
            cls.instance = super(AWSAccountClamp, cls).__new__(cls)

        # Return the singleton
        return cls.instance

    @classmethod
    def check_aws_identity(cls) -> bool:
        """Check the AWS Identity currently active"""
        # Check AWS Identity Token
        sts = boto3.client("sts")
        try:
            identity = sts.get_caller_identity()
            cls.log.info("AWS Account Info:")
            cls.log.info(f"Account: {identity['Account']}")
            cls.log.info(f"Identity ARN: {identity['Arn']}")
            cls.log.info(f"Region: {cls.region}")
            return True
        except (ClientError, UnauthorizedSSOTokenError):
            msg = "AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            cls.log.critical(msg)
            raise RuntimeError(msg)

    @classmethod
    def get_aws_account_info(cls) -> dict:
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
            info["Region"] = cls.region
            return info
        except (ClientError, UnauthorizedSSOTokenError):
            cls.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            return info

    @classmethod
    def check_s3_access(cls) -> bool:
        s3 = cls.boto3_session.client("s3")
        results = s3.list_buckets()
        for bucket in results["Buckets"]:
            cls.log.info(f"\t{bucket['Name']}")
        return True

    @classmethod
    def sagemaker_session(cls) -> "SageSession":
        """Create a sageworks SageMaker session (using our boto3 refreshable session)
        Args:
            session (boto3.Session, optional): A boto3 session to use. Defaults to None.
        Returns:
            SageSession: A SageMaker session object
        """
        from sagemaker.session import Session as SageSession

        return SageSession(boto_session=cls.boto3_session)

    @classmethod
    def sagemaker_client(cls) -> BaseClient:
        """Create a sageworks SageMaker client (using our boto3 refreshable session)
        Args:
            session (boto3.Session, optional): A boto3 session to use. Defaults to None.
        Returns:
            BaseClient: A SageMaker client object
        """
        return cls.boto3_session.client("sagemaker")


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
