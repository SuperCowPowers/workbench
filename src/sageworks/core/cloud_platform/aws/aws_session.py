import os
import sys

import boto3
import re
from botocore.exceptions import ClientError, UnauthorizedSSOTokenError, TokenRetrievalError, SSOTokenLoadError
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
import logging

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.ipython_utils import is_running_in_ipython, display_error_and_raise
from sageworks.utils.execution_environment import running_on_lambda, running_on_glue


class AWSSession:
    """AWSSession manages AWS Sessions and SageWorks Role Assumption"""

    def __init__(self):
        self.log = logging.getLogger("sageworks")

        # Pull in our config from the config manager
        self.cm = ConfigManager()
        self.sageworks_role_name = self.cm.get_config("SAGEWORKS_ROLE")

        # Grab the AWS Profile from the Config Manager
        profile = self.cm.get_config("AWS_PROFILE")
        if profile is not None:
            os.environ["AWS_PROFILE"] = profile

        # Grab our AWS Account Info
        try:
            self.account_id = boto3.client("sts").get_caller_identity()["Account"]
            self.region = boto3.Session().region_name
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError, SSOTokenLoadError):
            msg = "AWS SSO Token Failure: Check AWS_PROFILE and/or Renew SSO Token..."
            self.log.critical(msg)
            if is_running_in_ipython():
                display_error_and_raise(msg)
            else:
                sys.exit(1)

    @property
    def boto3_session(self):
        """Get the AWS Boto3 Session, defaulting to the SageWorks Role if possible."""

        # Check the execution environment and determine if we need to assume the SageWorks Role
        if running_on_lambda() or running_on_glue() or self.is_sageworks_role():
            self.log.important("Using the default Boto3 session...")
            return boto3.Session()

        # Okay, so we need to assume the SageWorks Role
        try:
            return self._sageworks_role_boto3_session()
        except Exception as e:
            msg = "Failed to Assume SageWorks Role: Check AWS_PROFILE and/or Renew SSO Token.."
            self.log.critical(msg)
            raise RuntimeError(msg) from e

    def is_sageworks_role(self) -> bool:
        """Helper: Check if the current AWS Identity is the SageWorks Role"""
        sts_client = boto3.client("sts")
        try:
            return self.sageworks_role_name in sts_client.get_caller_identity()["Arn"]
        except Exception as e:
            msg = f"Failed: get_caller_identity() for SageWorks Role: {e}"
            raise RuntimeError(msg)

    def get_sageworks_execution_role_arn(self):
        """Get the SageWorks Execution Role ARN"""
        # Validate the account ID is a 12-digit number
        if not self.account_id.isdigit() or len(self.account_id) != 12:
            raise ValueError("Invalid AWS account ID")

        # Validate the role name contains only allowed characters
        if not re.match(r"^[\w+=,.@-]+$", self.sageworks_role_name):
            raise ValueError("Invalid SageWorks role name")

        # Construct the ARN
        return f"arn:aws:iam::{self.account_id}:role/{self.sageworks_role_name}"

    def _sageworks_role_boto3_session(self):
        """Internal: Get a boto3 session with assumed SageWorks role and refreshing credentials"""
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self._assume_sageworks_role_session_credentials(),
            refresh_using=self._assume_sageworks_role_session_credentials,
            method="sts-assume-role",
        )
        session = get_session()
        session._credentials = refreshable_credentials
        return boto3.Session(botocore_session=session)

    def _assume_sageworks_role_session_credentials(self):
        """Internal: Assume SageWorks Role and set up AWS Session credentials with automatic refresh."""
        sts_client = boto3.client("sts")
        try:
            response = sts_client.assume_role(
                RoleArn=self.get_sageworks_execution_role_arn(),
                RoleSessionName="sageworks-execution-role-session",
            ).get("Credentials")

            credentials = {
                "access_key": response["AccessKeyId"],
                "secret_key": response["SecretAccessKey"],
                "token": response["SessionToken"],
                "expiry_time": response["Expiration"].isoformat(),
            }
            # Use direct print instead of logging due to deadlock concerns.
            self.c_print(f"AWS Credentials Refreshed: Expires at {response['Expiration'].astimezone()}")
            return credentials

        except Exception as e:
            self.c_print(f"Error during Refresh Credentials: {e}", critical=True)
            raise RuntimeError("Failed to refresh SageWorks role session credentials")

    @staticmethod
    def c_print(text: str, critical: bool = False):
        if critical:
            print(f"\x1b[38;5;198m{text}\x1b[0m")
        else:
            print(f"\x1b[38;5;69m{text}\x1b[0m")


if __name__ == "__main__":
    """Exercise the AWS Session Class"""

    my_aws_session = AWSSession()
    my_boto_session = AWSSession().boto3_session
    my_sts_client = my_boto_session.client("sts")

    print(f"Account ID: {my_sts_client.get_caller_identity()['Account']}")
    print(f"Region: {my_boto_session.region_name}")
    print(f"Identity: {my_sts_client.get_caller_identity()['Arn']}")
    print(f"Session Token: {my_boto_session.get_credentials().token}")
    print(f"Session Expiry: {my_boto_session.get_credentials()._expiry_time}")
    my_aws_session.c_print(f"Assumed Role: {my_boto_session.get_credentials().method}")
    my_aws_session.c_print("Fake Critical Message", critical=True)
