import logging
import os
import re
import shutil
import subprocess
import sys

import boto3
from botocore.credentials import RefreshableCredentials
from botocore.exceptions import ClientError, SSOTokenLoadError, TokenRetrievalError, UnauthorizedSSOTokenError
from botocore.session import get_session

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.utils.execution_environment import running_as_service
from workbench.utils.ipython_utils import display_error_and_raise, is_running_in_ipython


class AWSSession:
    """AWSSession (Singleton) manages AWS Sessions and Workbench Role Assumption"""

    _instance = None
    _cached_boto3_session = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AWSSession, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # Prevent reinitialization
            self.log = logging.getLogger("workbench")

            # Pull in our config from the config manager
            self.cm = ConfigManager()
            self.workbench_role_name = self.cm.get_config("WORKBENCH_ROLE")

            # Grab the AWS Profile from the Config Manager
            self.profile = self.cm.get_config("AWS_PROFILE") or os.environ.get("AWS_PROFILE")
            if self.profile is not None:
                os.environ["AWS_PROFILE"] = self.profile

            # Grab our AWS Account Info
            try:
                self._set_account_info()
            except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError, SSOTokenLoadError):
                msg = "AWS SSO Token Failure: Check AWS_PROFILE and/or Renew SSO Token..."
                self.log.critical(msg)
                if self.renew_sso_login():
                    try:
                        self._set_account_info()
                    except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError, SSOTokenLoadError):
                        self._sso_failure_exit(msg)
                else:
                    self._sso_failure_exit(msg)
            self.initialized = True  # Mark as initialized to prevent reinitialization

    def _set_account_info(self):
        """Set AWS account and region information from the active caller identity."""
        self.account_id = boto3.client("sts").get_caller_identity()["Account"]
        self.region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
        self.region = boto3.Session().region_name if self.region is None else self.region

    def renew_sso_login(self) -> bool:
        """Try to renew the configured AWS SSO profile through the AWS CLI."""
        if running_as_service() or not self.profile:
            return False

        if shutil.which("aws") is None:
            self.log.warning("AWS CLI not found; cannot open AWS SSO login automatically.")
            return False

        self.log.important(f"Opening AWS SSO login for profile '{self.profile}'...")
        try:
            result = subprocess.run(["aws", "sso", "login", "--profile", self.profile], check=False)
        except OSError as err:
            self.log.warning(f"Failed to launch AWS SSO login: {err}")
            return False

        if result.returncode != 0:
            self.log.warning(f"AWS SSO login failed for profile '{self.profile}' with exit code {result.returncode}.")
            return False
        return True

    @staticmethod
    def _sso_failure_exit(msg: str):
        """Exit or raise after SSO renewal has failed."""
        if is_running_in_ipython():
            display_error_and_raise(msg)
        else:
            sys.exit(1)

    @property
    def boto3_session(self):
        if self._cached_boto3_session is None:
            self._cached_boto3_session = self._create_boto3_session()
        return self._cached_boto3_session

    def _create_boto3_session(self):
        """Internal: Get the AWS Boto3 Session, assuming the Workbench Role if necessary."""

        # Check if we're running as a service or already using the Workbench Role
        if running_as_service() or self.is_workbench_role():
            self.log.important("Using the default Boto3 session...")
            return boto3.Session(region_name=self.region)

        # Okay, so we need to assume the Workbench Role
        try:
            return self._workbench_role_boto3_session()
        except Exception as e:
            msg = "Failed to Assume Workbench Role: Check AWS_PROFILE and/or Renew SSO Token.."
            self.log.critical(msg)
            raise RuntimeError(msg) from e

    def assumed_role_info(self) -> dict:
        """Get info about the assumed role by querying our internal boto3 session"""
        sts_client = self.boto3_session.client("sts")

        # Get the caller identity to verify the assumed role
        identity = sts_client.get_caller_identity()
        assumed_role_arn = identity["Arn"]
        account_id = identity["Account"]
        user_id = identity["UserId"]
        return {
            "AssumedRoleArn": assumed_role_arn,
            "AccountId": account_id,
            "UserId": user_id,
        }

    def is_workbench_role(self) -> bool:
        """Helper: Check if the current AWS Identity is the Workbench Role"""
        sts_client = boto3.client("sts")
        try:
            return self.workbench_role_name in sts_client.get_caller_identity()["Arn"]
        except Exception as e:
            msg = f"Failed: get_caller_identity() for Workbench Role: {e}"
            raise RuntimeError(msg)

    def get_workbench_execution_role_arn(self):
        """Get the Workbench Execution Role ARN"""
        # Validate the account ID is a 12-digit number
        if not self.account_id.isdigit() or len(self.account_id) != 12:
            raise ValueError("Invalid AWS account ID")

        # Validate the role name contains only allowed characters
        if not re.match(r"^[\w+=,.@-]+$", self.workbench_role_name):
            raise ValueError("Invalid Workbench role name")

        # Construct the ARN
        return f"arn:aws:iam::{self.account_id}:role/{self.workbench_role_name}"

    def _workbench_role_boto3_session(self):
        """Internal: Get a boto3 session with assumed Workbench role and refreshing credentials"""
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self._assume_workbench_role_session_credentials(),
            refresh_using=self._assume_workbench_role_session_credentials,
            method="sts-assume-role",
        )
        session = get_session()
        session._credentials = refreshable_credentials
        return boto3.Session(botocore_session=session)

    def _assume_workbench_role_session_credentials(self):
        """Internal: Assume Workbench Role and set up AWS Session credentials with automatic refresh."""
        sts_client = boto3.client("sts")
        try:
            response = sts_client.assume_role(
                RoleArn=self.get_workbench_execution_role_arn(),
                RoleSessionName="workbench-execution-role-session",
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
            raise RuntimeError("Failed to refresh Workbench role session credentials")

    @staticmethod
    def c_print(text: str, critical: bool = False):
        if critical:
            print(f"\x1b[38;5;198m{text}\x1b[0m")
        else:
            print(f"\x1b[38;5;69m{text}\x1b[0m")


if __name__ == "__main__":
    """Exercise the AWS Session Class"""

    # Print out info about the assumed role
    my_aws_session = AWSSession()
    my_aws_session.assumed_role_info()
