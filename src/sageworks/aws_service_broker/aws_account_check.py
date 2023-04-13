"""AWSAccountCheck runs a bunch of tests/checks to ensure SageWorks AWS Setup"""
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class AWSAccountCheck:
    """AWSAccountCheck runs a bunch of tests/checks to ensure SageWorks AWS Setup"""

    def __init__(self):
        """AWSAccountCheck Initialization"""
        self.log = logging.getLogger(__file__)

        # Create the AWSAccountClamp Class
        self.aws_clamp = AWSAccountClamp()

    def check(self):
        """Check if the AWS Account Clamp is 100% 'locked in'"""
        self.log.info("*** AWS Identity Check ***")
        self.aws_clamp.check_aws_identity()
        self.log.info("Identity Check Success...")

        self.log.info("*** AWS Assume SageWorks ExecutionRole Check ***")
        check_boto_session = self.aws_clamp.boto_session()
        self.log.info("Assume Role Success...")

        self.log.info("*** AWS App Config Check ***")
        self.aws_clamp.check_app_config(check_boto_session)
        self.log.info("App Config Check Success...")

        self.log.info("*** AWS S3 Access Check ***")
        self.aws_clamp.check_s3_access(check_boto_session)
        self.log.info("S3 Access Check Success...")

        self.log.info("*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.aws_clamp.sagemaker_client()
        self.log.info(sm_client.list_feature_groups()["FeatureGroupSummaries"])

        self.log.info("*** AWS Sagemaker Session/Client Check ***")
        sm_client = self.aws_clamp.sagemaker_client()
        self.log.info(sm_client.list_feature_groups()["FeatureGroupSummaries"])

        self.log.info("AWS Account Clamp: AOK!")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    aws_clamp = AWSAccountCheck()

    # Check that out AWS Account Clamp is working AOK
    aws_clamp.check()
