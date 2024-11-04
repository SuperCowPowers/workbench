"""Just a Utility Script that allows people to check which AWS Identity is active"""

import os
import sys
import logging

# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager


class AWSIdentityCheck:
    """Just a Utility Script that allows people to check which AWS Identity is active"""

    def __init__(self):
        """AWSIdentityCheck Initialization"""
        self.log = logging.getLogger("sageworks")

        # Create the AWSAccountClamp Class
        self.aws_clamp = AWSAccountClamp()

    def check(self):
        """Check the AWS Identity"""

        # Log if there's any AWS_PROFILE set
        cm = ConfigManager()
        active_profile = cm.get_config("AWS_PROFILE")
        if active_profile:
            self.log.info(f"SageWorks AWS_PROFILE: {active_profile}")
        else:
            self.log.info("No AWS_PROFILE set")
            sys.exit(0)

        # Now get the current AWS Identity from the AWSAccountClamp Class
        self.log.info("*** AWS Identity Check ***")
        self.aws_clamp.check_aws_identity()
        self.log.info("Identity Check Success...")


if __name__ == "__main__":
    """Exercise the AWS Account Clamp Class"""

    # Create the class
    aws_clamp = AWSIdentityCheck()

    # Check that out AWS Account Clamp is working AOK
    aws_clamp.check()
