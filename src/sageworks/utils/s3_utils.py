"""S3 Utilities for SageWorks"""

import os
import boto3
import logging

# Get the SageWorks Logger
from sageworks.utils import sageworks_logging  # noqa: F401

log = logging.getLogger("sageworks")


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


def copy_s3_files_to_local(s3_path: str, local_path: str):
    """Copies all files from S3 to a local directory, maintaining the subdirectory structure.
    Args:
        s3_path (str): S3 Path to the set of files (e.g., s3://bucket-name/path/to/files).
        local_path (str): Local directory to copy the files to.
    """
    s3_client = boto3.client("s3")
    bucket, key = s3_path.replace("s3://", "").split("/", 1)

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
            # Correctly handle object keys to ensure paths are relative
            relative_key = obj["Key"][len(key) :].lstrip("/")  # Remove the S3 prefix and leading slashes
            local_file_path = os.path.join(local_path, relative_key)

            # Ensure the subdirectory structure exists locally
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the object to the local file path
            log.important(f"Downloading {obj['Key']} to {local_file_path}")
            s3_client.download_file(bucket, obj["Key"], local_file_path)


if __name__ == "__main__":
    """Exercise the S3 Utilities"""
    import tempfile

    # Copy S3 files to local directory
    s3_path = "s3://sandbox-sageworks-artifacts/sageworks_plugins"
    local_path = tempfile.mkdtemp()
    copy_s3_files_to_local(s3_path, local_path)
