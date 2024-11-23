"""S3 Utilities for SageWorks"""

import os
import boto3
from botocore.exceptions import ClientError
import hashlib
import logging

# SageWorks imports
from sageworks.utils.performance_utils import performance

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


def ensure_s3_bucket_and_prefix(s3_uri: str, session: boto3.session.Session):
    """
    Ensure the S3 bucket and prefix exist, creating them if necessary.

    Args:
        s3_uri (str): The S3 URI (e.g., 's3://bucket-name/prefix/').
        session (boto3.session.Session): The boto3 session.
    """
    s3 = session.client("s3")

    # Parse bucket and prefix from the S3 path
    bucket, *prefix_parts = s3_uri.replace("s3://", "").split("/", 1)
    prefix = prefix_parts[0] if prefix_parts else ""

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Creating bucket: {bucket}")
            s3.create_bucket(Bucket=bucket)
        else:
            raise e

    # Ensure prefix exists by creating a placeholder object
    if prefix:
        print(f"Ensuring prefix: {prefix}")
        s3.put_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/.placeholder", Body=b"")


@performance
def compute_parquet_hash(s3_uri: str, session: boto3.session.Session) -> str:
    """
    Compute a hash for a set of Parquet files.

    Args:
        s3_uri (str): S3 URI for the FeatureGroup's offline storage (e.g., 's3://bucket-name/path/to/data/').
        session (boto3.session.Session): Boto3 session.

    Returns:
        str: Composite hash for a set of Parquet files
    """
    log = logging.getLogger("sageworks")
    s3 = session.client("s3")

    # Parse bucket and prefix from the S3 URI
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)

    # Ensure the prefix ends with a slash to match the exact directory
    if not prefix.endswith("/"):
        prefix += "/"

    # Initialize MD5 hash object
    md5_hash = hashlib.md5()

    # Use paginator to iterate through objects in the S3 prefix
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Ensure the file is directly under the desired prefix
            if obj["Key"].startswith(prefix) and obj["Key"].endswith(".parquet"):
                log.debug(f"Processing object: {obj['Key']}")
                etag = obj["ETag"].strip('"')  # Remove quotes around the ETag
                md5_hash.update(etag.encode("utf-8"))  # Add ETag to the composite hash

    return md5_hash.hexdigest()


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

    # Get our Account Clamp and S3 Bucket
    from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
    from sageworks.utils.config_manager import ConfigManager

    session = AWSAccountClamp().boto3_session
    sageworks_bucket = ConfigManager().get_config("SAGEWORKS_BUCKET")

    # Setup a temporary S3 prefix for the Athena output
    s3_scratch = f"s3://{sageworks_bucket}/temp/athena_output"

    # Check if a bucket and prefix exist
    print(f"Ensuring bucket and prefix exist: {s3_scratch}")
    ensure_s3_bucket_and_prefix(s3_scratch, session)

    # Copy S3 files to local directory
    """
    import tempfile
    s3_path = "s3://sandbox-sageworks-artifacts/sageworks_plugins"
    local_path = tempfile.mkdtemp()
    copy_s3_files_to_local(s3_path, local_path)
    """
