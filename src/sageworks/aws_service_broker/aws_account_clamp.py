"""AWSAccountClamp provides logic/functionality over a set of AWS IAM Services"""

import os

import boto3
import awswrangler as wr
from botocore.exceptions import (
    ClientError,
    UnauthorizedSSOTokenError,
    TokenRetrievalError,
)
from botocore.client import BaseClient
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
from sagemaker.session import Session as SageSession
from datetime import timedelta
import logging

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager, FatalConfigError


class AWSAccountClamp:
    def __new__(cls):
        """AWSAccountClamp Singleton Pattern"""
        if not hasattr(cls, "instance"):
            # Initialize class attributes here
            cls.log = logging.getLogger("sageworks")
            cls.log.info("Creating the AWSAccountClamp Singleton...")

            # Pull in our config from the config manager
            cls.cm = ConfigManager()
            if not cls.cm.config_okay():
                cls.log.error("SageWorks Configuration Incomplete...")
                cls.log.error("Run the 'sageworks' command and follow the prompts...")
                raise FatalConfigError()
            cls.role_name = cls.cm.get_config("SAGEWORKS_ROLE")
            cls.sageworks_bucket_name = cls.cm.get_config("SAGEWORKS_BUCKET")

            # Note: We might want to revisit this
            profile = cls.cm.get_config("AWS_PROFILE")
            if profile is not None:
                os.environ["AWS_PROFILE"] = profile

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

            # Initialize the boto3 session (this is a refreshable session)
            cls.session_time_delta = timedelta(minutes=50)
            cls.boto3_session = cls._init_boto3_session()
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
    def check_s3_access(cls, boto_session: boto3.Session) -> bool:
        s3 = boto_session.client("s3")
        results = s3.list_buckets()
        for bucket in results["Buckets"]:
            cls.log.info(f"\t{bucket['Name']}")
        return True

    @classmethod
    def ensure_aws_catalog_db(cls, catalog_db: str):
        """Ensure that the AWS Data Catalog Database exists"""
        cls.log.important(f"Ensuring that the AWS Data Catalog Database {catalog_db} exists...")
        wr.catalog.create_database(catalog_db, exist_ok=True, boto3_session=cls.boto3_session)

    @classmethod
    def is_sageworks_role(cls) -> bool:
        """Check if the current AWS Identity is the SageWorks Role"""
        sts = boto3.client("sts")
        try:
            if cls.role_name in sts.get_caller_identity()["Arn"]:
                return True
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError):
            msg = "SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            cls.log.critical(msg)
            raise RuntimeError(msg)

    @classmethod
    def sageworks_execution_role_arn(cls):
        """Get the SageWorks Execution Role ARN"""
        iam = boto3.client("iam")
        try:
            role_arn = iam.get_role(RoleName=cls.role_name)["Role"]["Arn"]
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            msg = f"Could Not Find Role {cls.role_name}"
            cls.log.critical(msg)
            raise RuntimeError(msg)
        except UnauthorizedSSOTokenError:
            msg = "SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            cls.log.critical(msg)
            raise RuntimeError(msg)

    @classmethod
    def glue_check(cls):
        """
        Check if the current execution environment is an AWS Glue job.

        Returns:
            bool: True if running in AWS Glue environment, False otherwise.
        """
        try:
            import awsglue  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def _session_credentials(cls):
        """Internal: Set up our AWS Session credentials for automatic refresh"""

        # Assume the SageWorks Execution Role and then pull the credentials
        cls.log.debug("Assuming the SageWorks Execution Role and Refreshing Credentials...")
        sts = boto3.Session().client("sts")
        response = sts.assume_role(
            RoleArn=cls.sageworks_execution_role_arn(),
            RoleSessionName="sageworks-execution-role-session",
        ).get("Credentials")
        credentials = {
            "access_key": response.get("AccessKeyId"),
            "secret_key": response.get("SecretAccessKey"),
            "token": response.get("SessionToken"),
            "expiry_time": response.get("Expiration").isoformat(),
        }
        cls.log.debug(f"Credentials Refreshed: Expires at {credentials['expiry_time']}")
        return credentials

    @classmethod
    def _init_boto3_session(cls):
        if cls.is_sageworks_role() or cls.glue_check():
            return boto3.Session()

        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=cls._session_credentials(),
            refresh_using=cls._session_credentials,
            method="sts-assume-role",
        )

        session = get_session()
        session._credentials = refreshable_credentials
        refreshable_session = boto3.Session(botocore_session=session)

        return refreshable_session

    @classmethod
    def boto_session(cls):
        """Create a *refreshable* AWS/boto session so that clients don't get TOKEN timeouts"""
        return cls.boto3_session

    @classmethod
    def sagemaker_session(cls, session: boto3.Session = None) -> SageSession:
        """Create a sageworks SageMaker session (using our boto3 refreshable session)
        Args:
            session (boto3.Session, optional): A boto3 session to use. Defaults to None.
        Returns:
            SageSession: A SageMaker session object
        """
        session = session or cls.boto_session()
        return SageSession(boto_session=session)

    @classmethod
    def sagemaker_client(cls, session: boto3.Session = None) -> BaseClient:
        """Create a sageworks SageMaker client (using our boto3 refreshable session)
        Args:
            session (boto3.Session, optional): A boto3 session to use. Defaults to None.
        Returns:
            BaseClient: A SageMaker client object
        """
        session = session or cls.boto_session()
        return session.client("sagemaker")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    aws_account_clamp = AWSAccountClamp()

    # Check out that AWS Account Clamp is working AOK
    """Check if the AWS Account is Setup Correctly"""
    print("*** AWS Identity Check ***")
    aws_account_clamp.check_aws_identity()
    print("Identity Check Success...")

    print("*** AWS Assume SageWorks ExecutionRole Check ***")
    check_boto_session = aws_account_clamp.boto_session()
    print("Assume Role Success...")

    print("*** AWS S3 Access Check ***")
    aws_account_clamp.check_s3_access(check_boto_session)
    print("S3 Access Check Success...")

    print("*** AWS Sagemaker Session/Client Check ***")
    sm_client = aws_account_clamp.sagemaker_client()
    print(sm_client.list_feature_groups()["FeatureGroupSummaries"])
