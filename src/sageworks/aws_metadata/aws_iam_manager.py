"""AWSIAMManager provides a bit of logic/functionality over the set of AWS IAM Services"""
import sys
import argparse
import logging

# Local Imports
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class AWSIAMManager:

    def __init__(self, database_scope='sageworks'):
        """"AWSIAMManager provides a bit of logic/functionality over the set of AWS IAM Services"""
        self.log = logging.getLogger(__file__)


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
