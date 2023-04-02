"""AWSAccountClamp provides logic/functionality over the set of AWS IAM Services"""
import sys
import boto3
from botocore.exceptions import ClientError, UnauthorizedSSOTokenError
from sagemaker.session import Session as SageSession
import argparse
import logging

# SageWorks Imports
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class AWSAccountClamp:
    def __init__(self, role_name="SageWorks-ExecutionRole"):
        """AWSAccountClamp provides logic/functionality over the set of AWS IAM Services"""
        self.log = logging.getLogger(__file__)
        self.role_name = role_name

    def check(self):
        """Check if the AWS Account Clamp is 100% 'locked in'
        - Check the current AWS Identity (Print it out)
        - See if we can 'assume-role' for the SageWorks-ExecutionRole
        - Test out S3 Access (with boto_session)
        - Test out SageMaker Access (with sm_session)
        - Test out SageMake Client (with sm_client)
        """
        self.log.info("\n*** AWS Identity Check ***")
        self.check_aws_identity()
        self.log.info("Identity Check Success...")

        self.log.info("\n*** AWS Assume SageWorks-ExecutionRole Check ***")
        my_boto_session = self.boto_session()
        self.log.info("Assume Role Success...")

        self.log.info("\n*** AWS S3 Access Check ***")
        s3 = my_boto_session.client("s3")
        results = s3.list_buckets()
        for bucket in results["Buckets"]:
            self.log.info(f"\t{bucket['Name']}")

        self.log.info("\n*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.sagemaker_client()
        self.log.info(sm_client.list_feature_groups()["FeatureGroupSummaries"])

        self.log.info("\n*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.sagemaker_client()
        self.log.info(sm_client.list_feature_groups()["FeatureGroupSummaries"])

        self.log.info("\nAWS Account Clamp: AOK!")

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

    def is_sageworks_role(self) -> bool:
        """Check if the current AWS Identity is the SageWorks Role"""
        sts = boto3.client("sts")
        try:
            if "SageWorks-ExecutionRole" in sts.get_caller_identity()["Arn"]:
                return True
        except (ClientError, UnauthorizedSSOTokenError) as exc:
            self.log.critical("SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it

    def sageworks_execution_role_arn(self):
        """Get the SageWorks Execution Role"""
        try:
            iam = boto3.client("iam")
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

    def boto_session(self):
        """Create a sageworks session using sts.assume_role(sageworks_execution_role)"""

        # First check if we have already assumed the SageWorks Execution Role
        if self.is_sageworks_role():
            return boto3.Session()

        # Okay we need to assume the SageWorks Execution Role and Return a boto3 session
        session = boto3.Session()
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=self.sageworks_execution_role_arn(),
            RoleSessionName="sageworks-execution-role-session",
        )
        new_session = boto3.Session(
            aws_access_key_id=response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
            aws_session_token=response["Credentials"]["SessionToken"],
        )
        return new_session

    def sagemaker_session(self):
        """Create a sageworks SageMaker session using sts.assume_role(sageworks_execution_role)"""
        return SageSession(boto_session=self.boto_session())

    def sagemaker_client(self):
        """Create a sageworks SageMaker client using sts.assume_role(sageworks_execution_role)"""
        return self.sagemaker_session().boto_session.client("sagemaker")

    @staticmethod
    def account_id():
        """Get the AWS AccountID"""
        return boto3.client("sts").get_caller_identity()["Account"]

    def region(self):
        """Get the AWS AccountID"""
        return self.boto_session().region_name


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create the class
    aws_clamp = AWSAccountClamp()

    # Check that out AWS Account Clamp is working AOK
    aws_clamp.check()
