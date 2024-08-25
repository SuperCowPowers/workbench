"""AWSAccountClamp provides logic/functionality over a set of AWS IAM Services"""

import os
import sys
import boto3
import watchtower
import getpass
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
from datetime import datetime
import logging

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager, FatalConfigError
from sageworks.utils.docker_utils import running_on_docker, running_on_ecs
from sageworks.utils.sageworks_logging import ColoredFormatter


class AWSAccountClamp:

    # Initialize Class Attributes
    log = None
    cm = None
    role_name = None
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

            # Check if this code is running in AWS Lambda, Glue, or already has the SageWorks Role
            cls.log.info("Checking Execution Environment...")
            if cls.running_on_lambda() or cls.running_on_glue() or cls.is_sageworks_role():
                cls.boto3_session = boto3.Session()
            else:
                # Assume the SageWorks Role and set up our AWS Session credentials with automatic refresh
                cls.boto3_session = cls._sageworks_role_boto3_session()

            # Add a Cloud Watch handler to the sageworks logger
            cls.log.info("Adding CloudWatch Logs Handler...")
            cls.log_stream_name = cls.determine_log_stream()
            cls.log.info(f"Log Stream Name: {cls.log_stream_name}")
            cls.add_cloudwatch_logs_handler()

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
            else:
                return False
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError):
            msg = "SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            cls.log.critical(msg)
            raise RuntimeError(msg)

    @classmethod
    def sageworks_execution_role_arn(cls):
        """Get the SageWorks Execution Role ARN"""
        return f"arn:aws:iam::{cls.account_id}:role/{cls.role_name}"

    @classmethod
    def running_on_glue(cls):
        """
        Check if the current execution environment is an AWS Glue job.

        Returns:
            bool: True if running in AWS Glue environment, False otherwise.
        """
        # Check if GLUE_VERSION or GLUE_PYTHON_VERSION is in the environment
        if "GLUE_VERSION" in os.environ or "GLUE_PYTHON_VERSION" in os.environ:
            cls.log.info("Running in AWS Glue Environment...")
            return True
        else:
            return False

    @classmethod
    def running_on_lambda(cls):
        """
        Check if the current execution environment is an AWS Lambda function.

        Returns:
            bool: True if running in AWS Lambda environment, False otherwise.
        """
        if "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
            cls.log.info("Running in AWS Lambda Environment...")
            return True
        else:
            return False

    @staticmethod
    def get_executable_name(argv):
        # Extract the script name from argv[0], get the base name, and remove the extension
        try:
            script_path = argv[0]
            base_name = os.path.basename(script_path)
            executable_name = os.path.splitext(base_name)[0]
            return executable_name
        except Exception:
            return None

    @classmethod
    def determine_log_stream(cls):
        """Determine the log stream name based on the environment."""
        executable_name = cls.get_executable_name(sys.argv)
        unique_id = cls.get_unique_identifier()

        if cls.running_on_lambda():
            job_name = executable_name or os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
            return f"lambda/{job_name}/{unique_id}"
        elif cls.running_on_glue():
            job_name = executable_name or os.environ.get("GLUE_JOB_NAME", "unknown")
            return f"glue/{job_name}/{unique_id}"
        elif running_on_ecs():
            job_name = executable_name or os.environ.get("ECS_TASK_DEFINITION_FAMILY", "unknown")
            return f"dashboard/{job_name}/{unique_id}"
        elif running_on_docker():
            job_name = executable_name or os.environ.get("SERVICE_NAME", "unknown")
            return f"docker/{job_name}/{unique_id}"
        else:
            # This should work across platforms, including Windows
            return f"laptop/{getpass.getuser()}"

    @classmethod
    def get_unique_identifier(cls):
        """Get a unique identifier for the log stream."""
        # Attempt to get the job ID from environment variables
        job_id = cls.get_job_id_from_environment()

        if job_id:
            return job_id
        else:
            # Fall back to a timestamp-based ID if no job ID is found
            return datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")

    @classmethod
    def get_job_id_from_environment(cls):
        """Try to retrieve the job ID from Glue or Lambda environment variables."""
        # Check for AWS Glue job ID
        glue_job_id = os.environ.get("GLUE_JOB_ID")
        if glue_job_id:
            return glue_job_id

        # Check for AWS Lambda request ID
        lambda_request_id = os.environ.get("AWS_LAMBDA_REQUEST_ID")
        if lambda_request_id:
            return lambda_request_id

        # If no job ID is found, return None
        return None

    @classmethod
    def add_cloudwatch_logs_handler(cls):
        """Add a CloudWatch Logs handler to the logger"""
        try:
            # Get the boto3 session from the SageWorks Account Clamp
            session = cls.boto3_session
            cloudwatch_client = session.client("logs")

            # Create a CloudWatch Logs handler
            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group="SageWorksLogGroup",
                stream_name=cls.log_stream_name,
                boto3_client=cloudwatch_client,
            )

            # Create a formatter for CloudWatch without the timestamp
            cloudwatch_formatter = ColoredFormatter("(%(filename)s:%(lineno)d) %(levelname)s %(message)s")
            cloudwatch_handler.setFormatter(cloudwatch_formatter)

            # Add the CloudWatch handler to the logger
            cls.log.addHandler(cloudwatch_handler)
        except ClientError as e:
            cls.log.error(f"Failed to set up CloudWatch Logs handler: {e}")

    @classmethod
    def _sageworks_role_boto3_session(cls):

        # Create a refreshable credentials object
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=cls._assume_sageworks_role_session_credentials(),
            refresh_using=cls._assume_sageworks_role_session_credentials,
            method="sts-assume-role",
        )
        session = get_session()
        session._credentials = refreshable_credentials
        refreshable_session = boto3.Session(botocore_session=session)

        return refreshable_session

    @classmethod
    def _assume_sageworks_role_session_credentials(cls):
        """Internal: Assume SageWorks Role and set up our AWS Session credentials for automatic refresh"""

        # Assume the SageWorks Execution Role and then pull the credentials
        cls.log.important("Assuming the SageWorks Execution Role with Refreshing Credentials...")
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
