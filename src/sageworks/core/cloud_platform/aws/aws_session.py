import os
import boto3
import re
from botocore.exceptions import ClientError, UnauthorizedSSOTokenError, TokenRetrievalError
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
import logging

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
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
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError) as e:
            self.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            raise RuntimeError("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...") from e

    @property
    def boto3_session(self):
        """Get the AWS Boto3 Session which might assume the SageWorks Role"""

        self.log.info("Checking Execution Environment...")
        if running_on_lambda() or running_on_glue() or self.is_sageworks_role():
            return boto3.Session()

        return self._sageworks_role_boto3_session()

    def is_sageworks_role(self) -> bool:
        """Check if the current AWS Identity is the SageWorks Role"""
        sts_client = boto3.client("sts")
        try:
            return self.sageworks_role_name in sts_client.get_caller_identity()["Arn"]
        except (ClientError, UnauthorizedSSOTokenError, TokenRetrievalError) as e:
            self.log.critical("SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            raise RuntimeError("SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token...") from e

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
        """Internal: Assume SageWorks Role and set up AWS Session credentials for automatic refresh"""

        self.log.info("Assuming the SageWorks Execution Role with Refreshing Credentials...")
        sts_client = boto3.client("sts")
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
        self.log.debug(f"Credentials Refreshed: Expires at {credentials['expiry_time']}")
        return credentials


if __name__ == "__main__":
    """Exercise the AWS Session Class"""

    my_session = AWSSession().boto3_session
    my_sts_client = my_session.client("sts")

    print(f"Account ID: {my_sts_client.get_caller_identity()['Account']}")
    print(f"Region: {my_session.region_name}")
    print(f"Identity: {my_sts_client.get_caller_identity()['Arn']}")
    print(f"Session Token: {my_session.get_credentials().token}")
    print(f"Session Expiry: {my_session.get_credentials()._expiry_time}")
    print(f"Assumed Role: {my_session.get_credentials().method}")
