"""S3 Utilities for Workbench"""

import os
import io
import json
import boto3
import awswrangler as wr
from botocore.exceptions import ClientError
import hashlib
from typing import Optional
import logging

# Workbench imports
from workbench.utils.performance_utils import performance

log = logging.getLogger("workbench")


def split_s3_path(s3_path: str) -> tuple[str, str]:
    """Split an S3 URI into its bucket and key/prefix.

    Args:
        s3_path (str): S3 URI (e.g., s3://bucket-name/path/to/files).

    Returns:
        tuple[str, str]: (bucket, key)

    Raises:
        ValueError: If not an s3:// URI, or it names no key/prefix (e.g., s3://bucket-name).
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {s3_path!r}")
    bucket, _, key = s3_path.removeprefix("s3://").partition("/")
    key = key.lstrip("/")  # "s3://bucket//key" addresses "key", not "/key"
    if not bucket or not key:
        raise ValueError(f"S3 path must include a bucket and a key/prefix: {s3_path!r}")
    return bucket, key


def upload_content_to_s3(content, path):
    """Write string content to S3 path"""
    bytes_io = io.BytesIO(content.encode("utf-8"))
    return wr.s3.upload(local_file=bytes_io, path=path)


def read_content_from_s3(path):
    """Read string content from S3 path"""
    buffer = io.BytesIO()
    wr.s3.download(path=path, local_file=buffer)
    buffer.seek(0)
    return buffer.read().decode("utf-8")


def read_s3_json(s3_uri: str, session: boto3.session.Session) -> Optional[dict]:
    """Read a JSON object from S3 into a dict (counterpart to inference.save_output).

    Args:
        s3_uri (str): The S3 URI of the object (e.g., 's3://bucket/key').
        session (boto3.session.Session): A boto3 session.

    Returns:
        Optional[dict]: The parsed JSON, or None if the object does not exist.
    """
    s3 = session.client("s3")
    bucket, key = split_s3_path(s3_uri)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return None


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
        bucket, key = split_s3_path(s3_uri)
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

    bucket_name, object_key = split_s3_path(s3_url)

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

    bucket_name, prefix = split_s3_path(s3_url)

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
    """Copies an S3 object or prefix to local, maintaining the subdirectory structure.

    Args:
        s3_path (str): S3 object or prefix (e.g., s3://bucket-name/path/to/files).
        local_path (str): Local destination. For a single object, an existing directory
            (or a trailing "/") means "into this directory"; otherwise it's the exact
            destination file. For a prefix, the directory to mirror into.
    """
    s3_client = boto3.client("s3")
    bucket, key = split_s3_path(s3_path)

    def _download(obj_key: str, dest: str):
        if os.path.dirname(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        log.important(f"Downloading {bucket}/{obj_key} to {dest}")
        s3_client.download_file(bucket, obj_key, dest)

    def _is_object(k: str) -> bool:
        # Only "it isn't there" means prefix; AccessDenied and friends must surface,
        # otherwise a permissions failure looks like an empty download.
        try:
            s3_client.head_object(Bucket=bucket, Key=k)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    if not key.endswith("/") and _is_object(key):
        into_dir = local_path.endswith(("/", os.sep)) or os.path.isdir(local_path)
        _download(key, os.path.join(local_path, os.path.basename(key)) if into_dir else local_path)
        return

    # Prefix: the trailing "/" keeps a sibling prefix (plugins_v2) from aliasing into plugins
    prefix = key.rstrip("/") + "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            relative_key = obj["Key"][len(prefix) :].lstrip("/")
            if not relative_key:  # the prefix's own placeholder object
                continue
            _download(obj["Key"], os.path.join(local_path, relative_key))


def copy_local_files_to_s3(local_path: str, s3_path: str):
    """Copies a local file or directory tree to S3, maintaining the subdirectory structure.

    Compiled Python artifacts (``__pycache__`` dirs and ``.pyc`` files) are skipped.

    Args:
        local_path (str): Local file or directory to copy.
        s3_path (str): S3 destination (e.g., s3://bucket-name/path/to/files). For a
            single file, a trailing "/" means "into this prefix"; otherwise it's the
            exact destination key.

    Raises:
        FileNotFoundError: If local_path doesn't exist.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Nothing to upload, path not found: {local_path}")
    s3_client = boto3.client("s3")
    bucket, key = split_s3_path(s3_path)

    def _upload(source: str, dest_key: str):
        log.important(f"Uploading {source} to {bucket}/{dest_key}")
        s3_client.upload_file(source, bucket, dest_key)

    if os.path.isfile(local_path):
        dest_key = f"{key.rstrip('/')}/{os.path.basename(local_path)}" if key.endswith("/") else key
        _upload(local_path, dest_key)
        return

    prefix = key.rstrip("/")
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if name.endswith(".pyc"):
                continue
            source = os.path.join(root, name)
            relative = os.path.relpath(source, local_path)
            _upload(source, f"{prefix}/{relative}")


if __name__ == "__main__":
    """Exercise the S3 Utilities"""

    # Get our Account Clamp and S3 Bucket
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
    from workbench.core.artifacts.aws_artifact import AWSArtifact

    session = AWSAccountClamp().boto3_session

    # Temporary S3 prefix under the shared scratch root (AWSArtifact.temp_s3_path)
    s3_scratch = f"{AWSArtifact.temp_s3_path}/s3_utils_test/"

    # Check if a bucket and prefix exist
    print(f"Ensuring bucket and prefix exist: {s3_scratch}")
    ensure_s3_bucket_and_prefix(s3_scratch, session)

    # Test the write and read functions
    test_string = "This is a test string."
    test_path = f"{s3_scratch}/test.txt"
    print(f"Writing to S3: {test_path}")
    upload_content_to_s3(test_string, test_path)
    print(f"Reading from S3: {test_path}")
    read_string = read_content_from_s3(test_path)
    print(f"Read string: {read_string}")
    assert read_string == test_string, "Read string does not match the original string."
