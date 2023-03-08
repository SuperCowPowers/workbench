"""AWSIAMManager provides a bit of logic/functionality over the set of AWS IAM Services"""
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


class AWSIAMManager:

    def __init__(self):
        """"AWSIAMManager provides a bit of logic/functionality over the set of AWS IAM Services"""
        self.log = logging.getLogger(__file__)

        # FIXME: Have some nice functionality around
        """
        - identity
        - roles
        - policies
        - Check whether a transform can 'create/update' the thing it's about to generate
        """

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

    def sagemaker_execution_role(self):
        # Grab our SageMaker Role
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            print('Setting SageMaker Role Explicitly... ')
            try:
                sm_role = 'AmazonSageMaker-ExecutionRole-20181215T180236'
                iam = boto3.client('iam')
                role = iam.get_role(RoleName=sm_role)['Role']['Arn']
                return role
            except iam.exceptions.NoSuchEntityException:
                print(f"Could Not Find Role {sm_role}")
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
    iam_manager = AWSIAMManager()

    # Check our AWS identity
    iam_manager.check_aws_identity()

    # Get our SageMaker Role
    my_role = iam_manager.sagemaker_execution_role()
    print(my_role)
