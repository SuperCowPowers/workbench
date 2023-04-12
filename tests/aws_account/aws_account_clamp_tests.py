"""Tests for the AWS Account Clamp"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_check import AWSAccountCheck


def test():
    """Tests for the AWS Account Clamp"""

    # Get the AWS Account Clamp and make sure it checks out
    aws_clamp = AWSAccountCheck()
    aws_clamp.check()


if __name__ == "__main__":
    test()
