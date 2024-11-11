import botocore
import logging
import sys
import time
import functools
import json
import base64
import re
import os
from typing import Union, List, Callable, Optional
import pandas as pd
import awswrangler as wr
from awswrangler.exceptions import NoFilesFound
from pathlib import Path
import posixpath
from botocore.exceptions import ClientError
from sagemaker.session import Session as SageSession
from sagemaker import image_uris
from collections.abc import Mapping, Iterable


# SageWorks Imports
from sageworks.utils.deprecated_utils import deprecated

# SageWorks Logger
log = logging.getLogger("sageworks")


def get_image_uri_with_digest(framework, region, version, sm_session: SageSession):
    # Retrieve the base image URI using sagemaker SDK
    base_image_uri = image_uris.retrieve(
        framework=framework, region=region, version=version, sagemaker_session=sm_session
    )
    print(f"Base Image URI: {base_image_uri}")

    # Extract repository name and image tag from the base image URI
    repo_uri, image_tag = base_image_uri.split(":")
    repository_name = repo_uri.split("/")[-1]

    # Use AWS CLI to get image details and find the digest
    ecr_client = sm_session.boto_session.client("ecr", region_name=region)
    response = ecr_client.describe_images(
        repositoryName=repository_name,
        imageIds=[
            {"imageTag": image_tag},
        ],
    )
    if "imageDetails" in response and len(response["imageDetails"]) > 0:
        image_digest = response["imageDetails"][0]["imageDigest"]
        full_image_uri = f"{repo_uri}@{image_digest}"
        return full_image_uri
    else:
        raise ValueError("Image details not found for the specified tag.")


def client_error_printout(err: botocore.exceptions.ClientError):
    """Helper method to get information about a botocore.exceptions.ClientError"""
    error_code = err.response["Error"]["Code"]
    error_message = err.response["Error"]["Message"]
    operation_name = err.operation_name
    service_name = err.response["ResponseMetadata"].get("ServiceName", "Unknown")
    request_id = err.response["ResponseMetadata"]["RequestId"]

    # Output the error information
    log.error(f"Error Code: {error_code}")
    log.error(f"Error Message: {error_message}")
    log.error(f"Operation Name: {operation_name}")
    log.error(f"Service Name: {service_name}")
    log.error(f"Request ID: {request_id}")


def aws_throttle(func=None, retry_intervals=None):
    """
    Decorator to handle AWS throttling exceptions with exponential backoff.

    Args:
        func: This is a decorator detail just ignore it.
        retry_intervals (list[int], optional): List of intervals in seconds between retries.
                                               If None, defaults to exponential backoff.
    """
    if func is None:
        return lambda f: aws_throttle(f, retry_intervals=retry_intervals)

    default_intervals = [2**i for i in range(1, 9)]  # Default exponential backoff: 2, 4, 8... 256 seconds
    intervals = retry_intervals or default_intervals

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt, delay in enumerate(intervals, start=1):
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ThrottlingException":
                    log_level = log.critical if delay > 100 else log.error if delay > 30 else log.warning
                    log_level(f"{func.__name__}: ThrottlingException ({attempt}): Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        # If we exhaust all retries, raise an exception
        raise ClientError(
            {
                "Error": {
                    "Code": "ThrottlingException",
                    "Message": f"{func.__name__} failed after {len(intervals)} retries",
                }
            },
            func.__name__,
        )

    return wrapper


def not_found_returns_none(func: Optional[Callable] = None, *, resource_name: str = "AWS resource") -> Callable:
    """Decorator to handle AWS resource not found (returns None) and re-raising otherwise.

    Args:
        func (Callable, optional): The function being decorated.
        resource_name (str): Name of the AWS resource being accessed. Used for clearer error messages.
    """
    not_found_errors = {
        "ResourceNotFound",
        "ResourceNotFoundException",
        "EntityNotFoundException",
        "ValidationException",
        "NoSuchBucket",
    }

    def decorator(inner_func: Callable) -> Callable:
        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            try:
                return inner_func(*args, **kwargs)
            except ClientError as error:
                error_code = error.response["Error"]["Code"]
                if error_code in not_found_errors:
                    log.warning(f"{resource_name} not found: {error_code}, returning None...")
                    return None
                else:
                    log.critical(f"Critical error in AWS call: {error_code}")
                    raise
            except wr.exceptions.NoFilesFound:
                log.info(f"Resource {resource_name} not found returning None...")
                return None

        return wrapper

    # If func is None, the decorator was called with arguments
    if func is None:
        return decorator
    else:
        # If func is not None, the decorator was used without arguments
        return decorator(func)


@deprecated(version="0.9")
def list_tags_with_throttle(arn: str, sm_session) -> dict:
    """A Wrapper around SageMaker's list_tags method that handles throttling
    Args:
        arn (str): The ARN of the SageMaker resource
        sm_session (SageSession): A SageMaker session object
    Returns:
        dict: A dictionary of tags

    Note: AWS List Tags can get grumpy if called too often
    """

    # Log the call
    log.debug(f"Calling list_tags for {arn}...")
    sleep_times = [2, 4, 8, 16, 32, 64]
    max_attempts = len(sleep_times)

    # Sanity check the ARN
    if arn is None:
        log.error("Called list_tags_with_throttle(arn==None)!")
        return {}

    # Loop with exponential backoff
    for attempt in range(max_attempts):
        try:
            # Call the AWS List Tags method
            aws_tags = sm_session.list_tags(arn)
            meta = aws_tags_to_dict(aws_tags)  # Convert the AWS Tags to a dictionary
            return meta

        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            # Check for ThrottlingException
            if error_code == "ThrottlingException":
                if attempt < 2:
                    log.info(f"ThrottlingException ({attempt}): list_tags on {arn}")
                elif attempt < 4:
                    log.warning(f"ThrottlingException ({attempt}): list_tags on {arn}")
                else:
                    log.error(f"ThrottlingException ({attempt}): list_tags on {arn}")

            # Check specific (Not Found) exceptions
            elif error_code in ["ValidationException", "ResourceNotFoundException"]:
                log.warning(f"Probably Fine -- {arn} AWS Validation/NotFound Exception: {error_code} - {error_message}")
                return {}
            else:
                # Handle other ClientErrors that may occur
                log.error(f"ClientError: {error_code} - {error_message}")
                raise e

            # Sleep for a bit before trying again
            time.sleep(sleep_times[attempt])

    # If we get here, we've failed to retrieve the tags
    log.error(f"Failed to retrieve tags for {arn}!")
    return {}


@deprecated(version="0.9")
def sagemaker_delete_tag(arn: str, sm_session: SageSession, key_to_remove: str):
    """Delete a tag from a SageMaker resource
    Args:
        arn (str): The ARN of the SageMaker resource
        sm_session (SageSession): A SageMaker session object
        key_to_remove (str): The metadata key to remove

    Note:
        Some tags might be 'chunked' into multiple tags, so we need to remove all of them
    """
    # Get the current tag keys
    current_keys = [tag["Key"] for tag in sm_session.list_tags(arn)]

    # Grab the client from our SageMaker session
    sm_client = sm_session.sagemaker_client

    # Check if this key is a regular tag
    if key_to_remove in current_keys:
        sm_client.delete_tags(ResourceArn=arn, TagKeys=[key_to_remove])

    # Check if this key is split into chunks
    else:
        keys_to_remove = []
        for key in current_keys:
            if key.startswith(f"{key_to_remove}_chunk_"):
                keys_to_remove.append(key)
        if keys_to_remove:
            log.info(f"Removing chunked tags {keys_to_remove}...")
            sm_client.delete_tags(ResourceArn=arn, TagKeys=keys_to_remove)


@deprecated(version="0.9")
def sageworks_meta_from_catalog_table_meta(table_meta: dict) -> dict:
    """Retrieve the SageWorks metadata from AWS Data Catalog table metadata
    Args:
        table_meta (dict): The AWS Data Catalog table metadata
    Returns:
        dict: The SageWorks metadata that's stored in the Parameters field
    """
    # Get the Parameters field from the table metadata
    params = table_meta.get("Parameters", {})
    return {key: decode_value(value) for key, value in params.items() if "sageworks" in key}


def decode_value(value):
    # Try to base64 decode the value
    try:
        value = base64.b64decode(value).decode("utf-8")
    except Exception:
        pass
    # Try to JSON decode the value
    try:
        value = json.loads(value)
    except Exception:
        pass

    # Okay, just return whatever we have
    return value


def is_valid_tag(tag):
    pattern = r"^([a-zA-Z0-9_.:/=+\-@]*)$"
    return re.match(pattern, tag) is not None


def dict_to_aws_tags(meta_data: dict) -> list:
    """AWS Tags are in an odd format, so we need to convert, encode, and chunk the data
    Args:
        meta_data (dict): Dictionary of metadata to convert to AWS Tags
    """
    chunked_data = {}  # Store any chunked data here
    chunked_keys = []  # Store any keys to remove here

    # AWS Tag Storage has the following constraints:
    # - 256 char maximum per tag
    # - 50 tags maximum per artifact (model/endpoint/etc)
    char_limit = 256
    max_tags = 50

    # Loop through the data: Convert non-string values to JSON strings, and chunk large data
    output_data = {}
    current_tags = len(meta_data.keys())
    for key, value in meta_data.items():
        # Convert data to JSON string
        if not isinstance(value, str):
            value = json.dumps(value, separators=(",", ":"))

        # Make sure the value is valid
        if not is_valid_tag(value):
            log.important(f"Base64 encoding metadata: {value}")
            value = base64.b64encode(value.encode()).decode()

        # Check if the value will fit in the 256-character limit
        if len(value) < char_limit:
            output_data[key] = value

        # If the value is longer than 256 but we have room to split it into chunks
        elif len(value) < (char_limit * (max_tags - current_tags)):
            log.important(f"Chunking metadata for key {key} with length {len(value)}...")
            chunked_keys.append(key)
            chunks = _chunk_data(key, value)
            for chunk in chunks:
                chunked_data[chunk] = chunks[chunk]
                current_tags += 1

        # Too long to store in AWS Tags
        else:
            log.error(f"Metadata for key '{key}' is quite big {len(value)} and shouldn't be stored in AWS Tags!")

    # Now add the chunked data to the output data
    output_data.update(chunked_data)
    log.info(f"Processed tags has {len(output_data.keys())} keys...")

    # Now convert to AWS Tags format
    aws_tags = []
    for key, value in output_data.items():
        aws_tags.append({"Key": key, "Value": value})
    return aws_tags


def aws_tags_to_dict(aws_tags) -> dict:
    """Internal: AWS Tags are in an odd format, so convert, decode, and de-chunk"""

    # Stitch together any chunked data
    stitched_data = {}
    regular_tags = {}

    for item in aws_tags:
        key = item["Key"]
        value = item["Value"]

        # Check if this key is a chunk
        if "_chunk_" in key:
            base_key, chunk_num = key.rsplit("_chunk_", 1)

            if base_key not in stitched_data:
                stitched_data[base_key] = {}

            stitched_data[base_key][int(chunk_num)] = value
        else:
            regular_tags[key] = decode_value(value)

    # Stitch chunks back together and decode
    for base_key, chunks in stitched_data.items():
        # Sort by chunk number and concatenate
        sorted_chunks = [chunks[i] for i in sorted(chunks.keys())]
        stitched_base64_str = "".join(sorted_chunks)

        # Decode the stitched base64 string
        try:
            stitched_json_str = base64.b64decode(stitched_base64_str).decode("utf-8")
        except UnicodeDecodeError:
            stitched_json_str = stitched_base64_str
        try:
            stitched_dict = json.loads(stitched_json_str)
        except json.decoder.JSONDecodeError:
            stitched_dict = stitched_json_str

        regular_tags[base_key] = stitched_dict

    return regular_tags


def _chunk_data(base_key: str, data: str) -> dict:
    # Initialize variables
    chunk_size = 256  # Max size for AWS tag value
    chunks = {}

    # Split the data into chunks with numbered 'chunk_' keys
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        chunks[f"{base_key}_chunk_{i // chunk_size + 1}"] = chunk

    return chunks


def extract_data_source_basename(source: str) -> str:
    """Extract the basename from a data source
    Args:
        source (str): The data source
    Returns:
        str: The basename of the data source
    """

    # If the source is a Pandas DataFrame, return 'dataframe'
    if isinstance(source, pd.DataFrame):
        return "dataframe"

    # Convert PosixPath to string if necessary
    if isinstance(source, Path):
        source = str(source)

    # Check if the source is an S3 path
    if source.startswith("s3://"):
        basename = posixpath.basename(source)
        name_without_extension = os.path.splitext(basename)[0]
        return name_without_extension + "_data"

    # Check if the source is a local file path
    elif os.path.isfile(source):
        basename = os.path.basename(source)
        name_without_extension = os.path.splitext(basename)[0]
        return name_without_extension + "_data"

    # If it's neither, assume it's already a data source name
    else:
        return source


def list_s3_files(s3_path: str, extensions: str = "*.csv") -> List[str]:
    """
    Lists files in an S3 path with specified extension.

    Args:
    s3_path (str): The full S3 path (e.g., 's3://my-bucket/my-prefix/').
    extensions (str): File extension to filter by, defaults to '*.csv'.

    Returns:
    List[str]: A list of file paths matching the extension in the S3 path.
    """
    files = wr.s3.list_objects(path=s3_path, suffix=extensions.lstrip("*"))
    return files


def newest_path(s3_locations: list[str], sm_session: SageSession) -> Union[str, None]:
    """Determine which S3 bucket and prefix combination has the newest files.

    Args:
        s3_locations (list[str]): A list of full S3 bucket and prefix combinations.
        sm_session (SageSession): A SageMaker session object.

    Returns:
        str: The full S3 bucket and prefix combination with the newest files, or None if no files are found.
    """
    s3_client = sm_session.boto_session.client("s3")
    newest_location, latest_time = None, None

    for location in s3_locations:
        bucket, prefix = location.replace("s3://", "", 1).split("/", 1)
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        for file in response.get("Contents", []):
            if latest_time is None or file["LastModified"] > latest_time:
                newest_location, latest_time = location, file["LastModified"]

    return newest_location


def pull_s3_data(s3_path: str, embedded_index=False) -> Union[pd.DataFrame, None]:
    """Helper method to pull data from S3 storage

    Args:
        s3_path (str): S3 Path to the Artifact
        embedded_index (bool, optional): Is the index embedded in the CSV? Defaults to False.

    Returns:
        pd.DataFrame: DataFrame of the Artifact (metrics, CM, regression_preds) (might be None)
    """

    # Sanity check for undefined S3 paths (None)
    if s3_path.startswith("None"):
        return None

    # Pull the CSV file from S3
    try:
        if embedded_index:
            df = wr.s3.read_csv(s3_path, index_col=0)
        else:
            df = wr.s3.read_csv(s3_path)
        return df
    except NoFilesFound:
        log.info(f"Could not find S3 data at {s3_path}...")
        return None
    except Exception as e:
        log.error(f"Failed to pull data from {s3_path}!")
        log.error(e)
        return None


@deprecated(version="0.9", stack_trace=True)
def compute_size(obj: object) -> int:
    """Recursively calculate the size of an object including its contents.

    Args:
        obj (object): The object whose size is to be computed.

    Returns:
        int: The total size of the object in bytes.
    """
    if isinstance(obj, Mapping):
        return sys.getsizeof(obj) + sum(compute_size(k) + compute_size(v) for k, v in obj.items())
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        return sys.getsizeof(obj) + sum(compute_size(item) for item in obj)
    else:
        return sys.getsizeof(obj)


@deprecated(version="0.9")
def num_columns_ds(data_info):
    """Helper: Compute the number of columns from the storage descriptor data"""
    try:
        return len(data_info["StorageDescriptor"]["Columns"])
    except KeyError:
        return "-"


@deprecated(version="0.9")
def num_columns_fs(data_info):
    """Helper: Compute the number of columns from the feature group data"""
    try:
        return len(data_info["FeatureDefinitions"])
    except KeyError:
        return "-"


@deprecated(version="0.9")
def aws_url(artifact_info, artifact_type, aws_account_clamp):
    """Helper: Try to extract the AWS URL from the Artifact Info Object"""
    if artifact_type == "S3":
        # Construct the AWS URL for the S3 Bucket
        name = artifact_info["Name"]
        region = aws_account_clamp.region
        s3_prefix = f"incoming-data/{name}"
        bucket_name = aws_account_clamp.sageworks_bucket_name
        base_url = "https://s3.console.aws.amazon.com/s3/object"
        return f"{base_url}/{bucket_name}?region={region}&prefix={s3_prefix}"
    elif artifact_type == "GlueJob":
        # Construct the AWS URL for the Glue Job
        region = aws_account_clamp.region
        job_name = artifact_info["Name"]
        base_url = f"https://{region}.console.aws.amazon.com/gluestudio/home"
        return f"{base_url}?region={region}#/editor/job/{job_name}/details"
    elif artifact_type == "DataSource":
        details = artifact_info.get("Parameters", {}).get("sageworks_details", "{}")
        return json.loads(details).get("aws_url", "unknown")
    elif artifact_type == "FeatureSet":
        aws_url = artifact_info.get("sageworks_meta", {}).get("aws_url", "unknown")
        # Hack for constraints on the SageMaker Feature Group Tags
        return aws_url.replace("__question__", "?").replace("__pound__", "#")


if __name__ == "__main__":
    """Exercise the AWS Utils"""
    from pprint import pprint
    from sageworks.core.artifacts.feature_set_core import FeatureSetCore
    from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    # Grab out SageMaker Session from the AWS Account Clamp
    sm_session = AWSAccountClamp().sagemaker_session()
    boto3_session = AWSAccountClamp().boto3_session

    # Get the Sagemaker client from the AWS Account Clamp
    sm_client = AWSAccountClamp().sagemaker_client()

    @aws_throttle
    def list_sagemaker_models():
        response = sm_client.list_models()
        return response["Models"]

    print(list_sagemaker_models())

    # Test the aws_throttle decorator with ThrottlingExceptions
    # @aws_throttle
    # def test_throttling():
    #     raise ClientError({"Error": {"Code": "ThrottlingException"}}, "test")
    # test_throttling()

    @not_found_returns_none
    def test_not_found():
        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "test")

    test_not_found()

    @not_found_returns_none(resource_name="my_not_found_resource")
    def test_not_found():
        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "test")

    test_not_found()

    try:

        @not_found_returns_none
        def test_other_error():
            # Raise a different error to test the error handler
            raise ClientError({"Error": {"Code": "SomeOtherError"}}, "test")

        test_other_error()
    except ClientError:
        print("AOK Expected Error :)")

    # Test a FeatureSetCore object (that uses some of these methods)
    my_features = FeatureSetCore("test_features")
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Get the image URI with digest
    framework = "sklearn"
    region = "us-west-2"
    version = "1.2-1"

    try:
        full_image_uri = get_image_uri_with_digest(framework, region, version, sm_session)
        print(f"Full Image URI with Digest: {full_image_uri}")
    except Exception as e:
        print(f"Error: {e}")

    # Test listing_files in an S3 folder method
    print(list_s3_files("s3://sageworks-public-data/common"))

    # Test the newest files in an S3 folder method
    s3_path = "s3://sandbox-sageworks-artifacts/endpoints/inference/abalone-regression-end"
    most_recent = newest_path([s3_path], sm_session)

    # Add a health tag
    my_features.add_health_tag("needs_onboard")
    print(my_features.get_health_tags())

    # Add a user tag
    my_features.add_tag("test_tag")
    my_tags = my_features.get_tags()
    pprint(my_tags)

    # Add sageworks meta data
    my_features.upsert_sageworks_meta({"test_meta": "test_value"})
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Test adding a None value
    my_features.upsert_sageworks_meta({"test_meta": None})
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Add sageworks meta data (testing regular expression constraints)
    my_features.upsert_sageworks_meta({"test_meta": "test_{:value"})
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Add non-string sageworks meta data
    my_features.upsert_sageworks_meta({"test_meta": {"foo": "bar"}})
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Add some large sageworks meta data (string)
    large_data = "x" * 512
    my_features.upsert_sageworks_meta({"large_meta": large_data})
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Add some large sageworks meta data (dict)
    large_data = {"data": "x" * 512, "more_data": "y" * 512}
    my_features.upsert_sageworks_meta({"large_meta": large_data})
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Remove the tag
    my_features.remove_sageworks_meta("large_meta")
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Test the list_tags_with_throttle method
    arn = my_features.arn()
    tags = list_tags_with_throttle(arn, sm_session)
    pprint(tags)
