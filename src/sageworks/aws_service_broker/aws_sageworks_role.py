"""AWSSageWorksRole provides a bit of logic/functionality over the set of AWS IAM Services"""
import sys
import boto3
from botocore.exceptions import ClientError
import sagemaker
import argparse
import logging

# Local Imports
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class AWSSageWorksRole:

    def __init__(self, role_name='SageWorks-ExecutionRole'):
        """"AWSSageWorksRole provides a bit of logic/functionality over the set of AWS IAM Services"""
        self.log = logging.getLogger(__file__)
        self.role_name = role_name

    def check_aws_identity(self) -> bool:
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

    def sageworks_session(self):
        """Create a sageworks session using sts.assume_role(sageworks_execution_role)"""
        # Make sure we can access stuff with this role
        session = boto3.Session()

        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=self.sageworks_execution_role_arn(),
            RoleSessionName="sageworks-session"
        )
        new_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                                    aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                                    aws_session_token=response['Credentials']['SessionToken'])
        return new_session

    def sageworks_execution_role_arn(self):
        """Get the SageWorks Execution Role"""
        try:
            iam = boto3.client('iam')
            role_arn = iam.get_role(RoleName=self.role_name)['Role']['Arn']
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            print(f"Could Not Find Role {self.role_name}")
            return None


if __name__ == '__main__':

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class
    iam_manager = AWSSageWorksRole()

    # Check our AWS identity
    iam_manager.check_aws_identity()

    # Get our SageWorks Session
    sageworks_session = iam_manager.sageworks_session()
    print(sageworks_session)

    # Try to access some AWS services
    s3 = sageworks_session.client("s3")
    print(s3.list_buckets())
