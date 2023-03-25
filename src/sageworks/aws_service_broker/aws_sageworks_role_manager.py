"""AWSSageWorksRoleManager provides a bit of logic/functionality over the set of AWS IAM Services"""
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


class AWSSageWorksRoleManager:

    def __init__(self, role_name='SageWorks-ExecutionRole'):
        """"AWSSageWorksRoleManagerSession: Get the SageWorks Execution Role and/or Session"""
        self.log = logging.getLogger(__file__)
        self.role_name = role_name

    def check_aws_identity(self) -> bool:
        """Check the AWS Identity currently active"""
        # Check AWS Identity Token
        sts = boto3.client('sts')
        try:
            identity = sts.get_caller_identity()
            self.log.info("\nAWS Account Info:")
            self.log.info(f"\tAccount: {identity['Account']}")
            self.log.info(f"\tARN: {identity['Arn']}")
            return True
        except (ClientError, UnauthorizedSSOTokenError) as exc:
            self.log.critical("AWS Identity Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it

    def is_sageworks_role(self) -> bool:
        """Check if the current AWS Identity is the SageWorks Role"""
        sts = boto3.client('sts')
        try:
            if 'SageWorks-ExecutionRole' in sts.get_caller_identity()['Arn']:
                return True
        except (ClientError, UnauthorizedSSOTokenError) as exc:
            self.log.critical("SageWorks Role Check Failure: Check AWS_PROFILE and/or Renew SSO Token...")
            self.log.critical(exc)
            sys.exit(1)  # FIXME: Longer term we probably want to raise exc and have caller catch it

    def sageworks_execution_role_arn(self):
        """Get the SageWorks Execution Role"""
        try:
            iam = boto3.client('iam')
            role_arn = iam.get_role(RoleName=self.role_name)['Role']['Arn']
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
            RoleSessionName="sageworks-boto3-session"
        )
        new_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                                    aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                                    aws_session_token=response['Credentials']['SessionToken'])
        return new_session

    def sagemaker_session(self):
        """Create a sageworks SageMaker session using sts.assume_role(sageworks_execution_role)"""
        # Make sure we can access stuff with this role
        return SageSession(boto_session=self.boto_session())


if __name__ == '__main__':

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class
    sageworks_role = AWSSageWorksRoleManager()

    # Check our AWS identity
    sageworks_role.check_aws_identity()

    # Get our Execution Role ARN
    role_arn = sageworks_role.sageworks_execution_role_arn()
    print(role_arn)

    # Get our Boto Session
    boto_session = sageworks_role.boto_session()
    print(boto_session)

    # Try to access a 'regular' AWS service
    s3 = boto_session.client("s3")
    print(s3.list_buckets())

    # Get our SageMaker Session
    sagemaker_session = sageworks_role.sagemaker_session()
    print(sagemaker_session)

    # Try to access a SageMaker AWS services
    sm_client = sagemaker_session.boto_session.client("sagemaker")
    print(sm_client.list_feature_groups()['FeatureGroupSummaries'])
