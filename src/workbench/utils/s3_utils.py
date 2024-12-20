"""S3 Utilities for Workbench"""

import os
import boto3
from urllib.parse import urlparse
from botocore.exceptions import ClientError
import hashlib
from typing import Optional
import logging

# Workbench imports
from workbench.utils.performance_utils import performance

log = logging.getLogger("workbench")


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


def get_s3_etag(s3_uri: str, session: boto3.session.Session) -> Optional[str]:
    """
    Retrieve the ETag of an S3 object.

    Args:
        s3_uri (str): The S3 URI of the object (e.g., 's3://bucket/key').
        session (boto3.session.Session): A boto3 session.

    Returns:
        Optional[str]: The ETag of the object if it exists, otherwise None.

    Note:
        In general AWS ETags aren't useful, they aren't content hashes, they just indicate
        'change' in the object, and for that you can just use the last modified date.
    """
    s3 = session.client("s3")

    try:
        # Parse bucket and key from the S3 URI
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        response = s3.head_object(Bucket=bucket, Key=key)
        return response.get("ETag", "").strip('"')  # Remove quotes from ETag
    except s3.exceptions.ClientError:
        return None


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


def compute_s3_object_hash(s3_url: str, session: boto3.session.Session) -> str:
    """
    Compute the MD5 hash of an S3 object's content.

    Args:
        s3_url (str): The S3 URL (e.g., "s3://bucket-name/object-key").
        session (boto3.session.Session): Boto3 session.

    Returns:
        str: MD5 hash of the object's content.
    """
    log.important(f"Computing S3 Object Hash: {s3_url}")

    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip("/")

    s3_client = session.client("s3")
    file_hash = hashlib.md5()
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)

    # Stream the object content to avoid memory overhead
    for chunk in response["Body"].iter_chunks(chunk_size=8192):
        file_hash.update(chunk)

    return file_hash.hexdigest()


@performance
def compute_parquet_hash(s3_url: str, session: boto3.session.Session) -> str:
    """
    Compute a composite content hash for a set of Parquet files in an S3 prefix.

    Args:
        s3_url (str): S3 URL for the FeatureGroup's offline storage (e.g., "s3://bucket-name/path/to/data/").
        session (boto3.session.Session): Boto3 session.

    Returns:
        str: Composite hash for a set of Parquet files.
    """
    log = logging.getLogger("workbench")
    s3_client = session.client("s3")

    # Parse bucket and prefix from the S3 URL
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path.lstrip("/")

    # Ensure the prefix ends with a slash to match the exact directory
    if not prefix.endswith("/"):
        prefix += "/"

    # Initialize MD5 hash object for composite hash
    composite_hash = hashlib.md5()

    # Use paginator to iterate through objects in the S3 prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].startswith(prefix) and obj["Key"].endswith(".parquet"):
                # Compute the hash for the current Parquet file
                file_hash = compute_s3_object_hash(f"s3://{bucket_name}/{obj['Key']}", session)
                composite_hash.update(file_hash.encode("utf-8"))
                log.debug(f"Hash for {obj['Key']}: {file_hash}")

    return composite_hash.hexdigest()


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
            log.important(f"Downloading {bucket}/{obj['Key']} to {local_file_path}")
            s3_client.download_file(bucket, obj["Key"], local_file_path)


if __name__ == "__main__":
    """Exercise the S3 Utilities"""

    # Get our Account Clamp and S3 Bucket
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
    from workbench.utils.config_manager import ConfigManager

    session = AWSAccountClamp().boto3_session
    workbench_bucket = ConfigManager().get_config("WORKBENCH_BUCKET")

    # Setup a temporary S3 prefix for the Athena output
    s3_scratch = f"s3://{workbench_bucket}/temp/athena_output"

    # Check if a bucket and prefix exist
    print(f"Ensuring bucket and prefix exist: {s3_scratch}")
    ensure_s3_bucket_and_prefix(s3_scratch, session)

    # Copy S3 files to local directory
    """
    import tempfile
    s3_path = "s3://sandbox-workbench-artifacts/workbench_plugins"
    local_path = tempfile.mkdtemp()
    copy_s3_files_to_local(s3_path, local_path)
    """
