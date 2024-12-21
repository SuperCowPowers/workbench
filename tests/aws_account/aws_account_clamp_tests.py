"""Tests for the AWS Account Clamp"""

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


def test():
    """Tests for the AWS Account Clamp"""

    # Create the class
    aws_account_clamp = AWSAccountClamp()

    # Check out that AWS Account Clamp is working AOK
    """Check if the AWS Account is Setup Correctly"""
    print("*** AWS Caller/Base Identity Check ***")
    aws_account_clamp.check_aws_identity()
    print("Caller/Base Identity Check Success...")

    print("*** AWS Assumed Role Check ***")
    aws_account_clamp.check_assumed_role()

    print("*** AWS S3 Access Check ***")
    aws_account_clamp.check_s3_access()
    print("S3 Access Check Success...")

    print("*** AWS Sagemaker Session/Client Check ***")
    sm_client = aws_account_clamp.sagemaker_client()
    print(sm_client.list_feature_groups()["FeatureGroupSummaries"])
