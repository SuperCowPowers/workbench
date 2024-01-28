"""S3 Utilities for SageWorks"""

import boto3


def read_s3_file(s3_path: str) -> str:
    """Reads a file from S3 and returns its content as a string
    Args:
        s3_path (str): S3 Path to the file
    Returns:
        str: Contents of the file as a string
    """
    s3_client = boto3.client("s3")
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")
