"""AWSSageWorksRoleManager provides a bit of logic/functionality over the set of AWS IAM Services"""
import sys
import boto3
from botocore.exceptions import ClientError
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

    @staticmethod
    def check_aws_identity() -> bool:
        """Check the AWS Identity currently active"""
        # Check AWS Identity Token
        sts = boto3.client('sts')
        try:
            identity = sts.get_caller_identity()
            print("\nAWS Account Info:")
            print(f"\tAccount: {identity['Account']}")
            print(f"\tARN: {identity['Arn']}")
            return True
        except ClientError:
            print("AWS Identity Check Failure: Check AWS_PROFILE and Renew Security Token...")
            return False

    def sageworks_execution_role_arn(self):
        """Get the SageWorks Execution Role"""
        try:
            iam = boto3.client('iam')
            role_arn = iam.get_role(RoleName=self.role_name)['Role']['Arn']
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            print(f"Could Not Find Role {self.role_name}")
            return None

    def boto_session(self):
        """Create a sageworks session using sts.assume_role(sageworks_execution_role)"""
        # Make sure we can access stuff with this role
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
