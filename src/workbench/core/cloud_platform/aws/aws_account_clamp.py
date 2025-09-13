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

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_session import AWSSession
from workbench.utils.config_manager import ConfigManager, FatalConfigError


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
        self.log = logging.getLogger("workbench")
        self.log.info("Creating the AWSAccountClamp Singleton...")

        # ConfigManager and AWS setup
        self.cm = ConfigManager()
        if not self.cm.config_okay():
            self.log.error("Workbench Configuration Incomplete...")
            self.log.error("Run the 'workbench' command and follow the prompts...")
            raise FatalConfigError()
        self.workbench_bucket_name = self.cm.get_config("WORKBENCH_BUCKET")
        self.aws_session = AWSSession()
        self.boto3_session = self.aws_session.boto3_session

        # Check our caller/base AWS Identity
        try:
            self.account_id = boto3.client("sts").get_caller_identity()["Account"]
            self.region = boto3.session.Session().region_name
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError):
            self.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            raise FatalConfigError()

        # Check our Assume Role
        self.log.info("Checking Workbench Assumed Role...")
        role_info = self.aws_session.assumed_role_info()
        self.log.info(f"Assumed Role: {role_info}")

        # Check if we have tag write permissions (if we don't, we are read-only)
        self.read_only = not self.check_tag_permissions()

        # Check our Workbench API Key and Load the License
        self.log.info("Checking Workbench API License...")
        self.cm.load_and_check_license(self.account_id)
        self.cm.print_license_info()

        # Mark the instance as initialized
        self._initialized = True

    def check_aws_identity(self) -> bool:
        """Check the Caller/Base AWS Identity currently active (not the assumed role)"""

        # Using the caller/base boto3 client (not the assumed role session)
        try:
            sts = boto3.client("sts")
            sts.get_caller_identity()
            return True
        except (ClientError, UnauthorizedSSOTokenError):
            msg = "AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            self.log.critical(msg)
            raise RuntimeError(msg)

    def check_assumed_role(self) -> bool:
        """Check the AWS Identity of the Assumed Role"""
        try:
            self.aws_session.assumed_role_info()
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

    def check_workbench_bucket(self) -> bool:
        """Check if the Workbench S3 Bucket exists and is accessible"""
        s3 = self.boto3_session.client("s3")  # Use client, not resource
        try:
            s3.head_bucket(Bucket=self.workbench_bucket_name)
            self.log.info(f"The {self.workbench_bucket_name} bucket exists and is accessible")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                self.log.critical(f"The {self.workbench_bucket_name} bucket does not exist")
            elif error_code == "403":
                self.log.critical(f"Access denied to {self.workbench_bucket_name} bucket")
            else:
                self.log.error(f"Error checking S3 bucket: {e}")
            return False

    def sagemaker_session(self) -> "SageSession":
        """Create a workbench SageMaker session (using our boto3 refreshable session)

        Returns:
            SageSession: A SageMaker session object
        """
        return SageSession(boto_session=self.boto3_session)

    def sagemaker_client(self) -> BaseClient:
        """Create a workbench SageMaker client (using our boto3 refreshable session)

        Returns:
            BaseClient: A SageMaker client object
        """
        return self.boto3_session.client("sagemaker")

    def check_tag_permissions(self):
        """Check if current role has permission to add tags to SageMaker endpoints.

        Returns:
            bool: True if AddTags is allowed, False otherwise
        """
        try:
            sagemaker = self.boto3_session.client("sagemaker")

            # Use a non-existent endpoint name
            fake_endpoint = "workbench-permission-check-dummy-endpoint"

            # Try to add tags to the non-existent endpoint
            sagemaker.add_tags(
                ResourceArn=f"arn:aws:sagemaker:{self.region}:{self.account_id}:endpoint/{fake_endpoint}",
                Tags=[{"Key": "PermissionCheck", "Value": "Test"}],
            )

            # If we get here, we have permission (but endpoint doesn't exist)
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            # AccessDeniedException = no permission
            if error_code == "AccessDeniedException":
                self.log.debug("No AddTags permission (AccessDeniedException)")
                return False

            # ResourceNotFound = we have permission, but endpoint doesn't exist
            elif error_code in ["ResourceNotFound", "ValidationException"]:
                self.log.debug("AddTags permission verified (resource not found)")
                return True

            # Unexpected error, assume no permission for safety
            else:
                self.log.debug(f"Unexpected error checking permissions: {error_code}")
                return False


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    aws_account_clamp = AWSAccountClamp()

    # Check out that AWS Account Clamp is working AOK
    """Check if the AWS Account is Setup Correctly"""
    print("\n\n*** AWS Caller/Base Identity Check ***")
    aws_account_clamp.check_aws_identity()
    print("Caller/Base Identity Check Success...")

    print("\n\n*** AWS Assumed Role Check ***")
    aws_account_clamp.check_assumed_role()
    print("Assumed Role Check Success...")

    print("\n\n*** AWS Workbench Bucket Check ***")
    aws_account_clamp.check_workbench_bucket()
    print("Workbench Bucket Check Success...")

    print("\n\n*** AWS Sagemaker Session/Client Check ***")
    sm_client = aws_account_clamp.sagemaker_client()
    print(sm_client.list_feature_groups()["FeatureGroupSummaries"])

    print("\n\n*** AWS Tag Permission Check ***")
    if aws_account_clamp.check_tag_permissions():
        print("Tag Permission Check Success...")
    else:
        print("Tag Permission Check Failed...")
