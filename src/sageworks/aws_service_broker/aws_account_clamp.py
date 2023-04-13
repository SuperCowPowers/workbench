"""AWSAccountClamp provides logic/functionality over a set of AWS IAM Services"""
import sys
import boto3
from botocore.exceptions import ClientError, UnauthorizedSSOTokenError
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
from sagemaker.session import Session as SageSession
from datetime import timedelta
import logging

# SageWorks Imports
from sageworks.utils.sageworks_config import SageWorksConfig
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class AWSAccountClamp:
    def __init__(self):
        """AWSAccountClamp provides logic/functionality over a set of AWS IAM Services"""
        self.log = logging.getLogger(__file__)

        # Grab the AWS Role Name from the SageWorks Config
        config = SageWorksConfig()
        role_name = config.get_config_value("SAGEWORKS_AWS", "SAGEWORKS_ROLE")
        self.role_name = role_name

        # The default AWS Assume Role TTL is 1 hour, so we'll set our TTL to 50 minutes
        self.session_time_delta = timedelta(minutes=50)

    def check_aws_identity(self) -> bool:
        """Check the AWS Identity currently active"""
        # Check AWS Identity Token
        sts = boto3.client("sts")
        try:
            identity = sts.get_caller_identity()
            self.log.info("AWS Account Info:")
            self.log.info(f"Account: {identity['Account']}")
            self.log.info(f"ARN: {identity['Arn']}")
            self.log.info(f"Region: {self.region()}")
            return True
        except (ClientError, UnauthorizedSSOTokenError) as exc:
            self.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it

    def check_app_config(self, boto_session: boto3.Session) -> bool:
        """Check if the AWS AppConfig Service is enabled"""
        # FIXME: This will be enabled later
        return True
        appconfig = boto_session.client("appconfig")
        try:
            appconfig.list_applications()
            return True
        except (ClientError, UnauthorizedSSOTokenError) as exc:
            self.log.critical("AWS AppConfig Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)

    def check_s3_access(self, boto_session: boto3.Session) -> bool:
        s3 = boto_session.client("s3")
        results = s3.list_buckets()
        for bucket in results["Buckets"]:
            self.log.info(f"\t{bucket['Name']}")
        return True

    def is_sageworks_role(self) -> bool:
        """Check if the current AWS Identity is the SageWorks Role"""
        sts = boto3.client("sts")
        try:
            if self.role_name in sts.get_caller_identity()["Arn"]:
                return True
        except (ClientError, UnauthorizedSSOTokenError) as exc:
            self.log.critical("SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it

    def sageworks_execution_role_arn(self):
        """Get the SageWorks Execution Role"""
        iam = boto3.client("iam")
        try:
            role_arn = iam.get_role(RoleName=self.role_name)["Role"]["Arn"]
            return role_arn
        except iam.exceptions.NoSuchEntityException as exc:
            self.log.critical(f"Could Not Find Role {self.role_name}")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it
        except UnauthorizedSSOTokenError as exc:
            self.log.critical("SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it

    def _session_credentials(self):
        """Internal: Set up our AWS Session credentials for automatic refresh"""

        # Assume the SageWorks Execution Role and then pull the credentials
        self.log.info("Assuming the SageWorks Execution Role and Refreshing Credentials...")
        sts = boto3.Session().client("sts")
        response = sts.assume_role(
            RoleArn=self.sageworks_execution_role_arn(),
            RoleSessionName="sageworks-execution-role-session",
        ).get("Credentials")
        credentials = {
            "access_key": response.get("AccessKeyId"),
            "secret_key": response.get("SecretAccessKey"),
            "token": response.get("SessionToken"),
            "expiry_time": response.get("Expiration").isoformat(),
        }
        self.log.info(f"Credentials Refreshed: Expires at {credentials['expiry_time']}")
        return credentials

    def boto_session(self):
        """Create a *refreshable* AWS/boto session so that clients don't get TOKEN timeouts"""

        # If we're already using the SageWorks Execution Role, then we don't need refreshable credentials
        if self.is_sageworks_role():
            return boto3.Session()

        # Get our refreshable credentials
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self._session_credentials(),
            refresh_using=self._session_credentials,
            method="sts-assume-role",
        )

        # Attach the refreshable credentials to a generic boto3 session
        session = get_session()
        session._credentials = refreshable_credentials

        # Create a new boto3 session using the refreshable credentials
        refreshable_session = boto3.Session(botocore_session=session)
        return refreshable_session

    def sagemaker_session(self, session: boto3.Session = None):
        """Create a sageworks SageMaker session (using our boto3 refreshable session)"""
        session = session or self.boto_session()
        return SageSession(boto_session=session)

    def sagemaker_client(self, session: boto3.Session = None):
        """Create a sageworks SageMaker client (using our boto3 refreshable session)"""
        session = session or self.boto_session()
        return session.client("sagemaker")

    @staticmethod
    def account_id():
        """Get the AWS AccountID"""
        return boto3.client("sts").get_caller_identity()["Account"]

    def region(self):
        """Get the AWS AccountID"""
        return self.boto_session().region_name


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Import the Check Class to test this class :)
    from sageworks.aws_service_broker.aws_account_check import AWSAccountCheck

    # Create the class
    aws_account_check = AWSAccountCheck()

    # Check that out AWS Account Clamp is working AOK
    aws_account_check.check()
