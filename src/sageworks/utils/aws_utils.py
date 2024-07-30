import botocore
import logging
import sys
import time
import json
import base64
import re
import os
from typing import Union
import pandas as pd
import awswrangler as wr
from awswrangler.exceptions import NoFilesFound
from pathlib import Path
import posixpath
from sagemaker.session import Session as SageSession
from sagemaker import image_uris
from collections.abc import Mapping, Iterable

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


def client_error_info(err: botocore.exceptions.ClientError):
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


def list_tags_with_throttle(arn: str, sm_session: SageSession) -> dict:
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
    sleep_time = 0.25

    # Loop 4 times with exponential backoff
    for i in range(4):
        try:
            # Call the AWS List Tags method
            aws_tags = sm_session.list_tags(arn)
            meta = _aws_tags_to_dict(aws_tags)  # Convert the AWS Tags to a dictionary
            return meta

        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                log.info(f"ThrottlingException: list_tags on {arn}")
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                # Handle other ClientErrors that may occur
                log.error(f"Unexpected ClientError: {error_code}")
                raise e

    # If we get here, we've failed to retrieve the tags
    log.error(f"Failed to retrieve tags for {arn}!")
    return {}


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


def _aws_tags_to_dict(aws_tags) -> dict:
    """Internal: AWS Tags are in an odd format, so convert to regular dictionary"""

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


def is_valid_tag(tag):
    pattern = r"^([a-zA-Z0-9_.:/=+\-@]*)$"
    return re.match(pattern, tag) is not None


def dict_to_aws_tags(meta_data: dict) -> list:
    """AWS Tags are in an odd format, so we need to convert data into the AWS Tag format
    Args:
        meta_data (dict): Dictionary of metadata to convert to AWS Tags
    """
    chunked_data = {}  # Store any chunked data here
    chunked_keys = []  # Store any keys to remove here

    # Loop through the data: Convert non-string values to JSON strings, and chunk large data
    for key, value in meta_data.items():
        # Convert data to JSON string
        if not isinstance(value, str):
            value = json.dumps(value, separators=(",", ":"))

        # Make sure the value is valid
        if not is_valid_tag(value):
            log.important(f"Base64 encoding metadata: {value}")
            value = base64.b64encode(value.encode()).decode()

        # Check if the value will fit in the 256-character limit
        if len(value) < 256:
            meta_data[key] = value

        # If the value is longer than 256 but less than 4096, split it into chunks
        elif len(value) < 4096:
            log.important(f"Chunking metadata for key {key} with length {len(value)}...")
            chunked_keys.append(key)
            chunks = _chunk_data(key, value)
            for chunk in chunks:
                chunked_data[chunk] = chunks[chunk]

        # Too long to store in AWS Tags
        else:
            log.error(f"Metadata for key {key} is too long to store in AWS Tags!")

    # Now remove any keys that were chunked and add the chunked data
    for key in chunked_keys:
        del meta_data[key]
    meta_data.update(chunked_data)

    # Now convert to AWS Tags format
    aws_tags = []
    for key, value in meta_data.items():
        aws_tags.append({"Key": key, "Value": value})
    return aws_tags


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


def newest_files(s3_locations: list[str], sm_session: SageSession) -> Union[str, None]:
    """Determine which full S3 bucket and prefix combination has the newest files.

    Args:
        s3_locations (list[str]): A list of full S3 bucket and prefix combinations.
        sm_session (SageSession): A SageMaker session object.

    Returns:
        str: The full S3 bucket and prefix combination with the newest files, or None if no files are found.
    """
    # Get the S3 client
    s3_client = sm_session.boto_session.client("s3")

    newest_location = None
    latest_time = None

    for location in s3_locations:
        # Extract bucket and prefix from the location
        bucket, prefix = location.replace("s3://", "").split("/", 1)

        # List files under the current prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        # Check if there are files
        if "Contents" in response:
            # Find the latest file in the current location
            for file in response["Contents"]:
                if newest_location is None or file["LastModified"] > latest_time:
                    newest_location = location
                    latest_time = file["LastModified"]

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


def num_columns_ds(data_info):
    """Helper: Compute the number of columns from the storage descriptor data"""
    try:
        return len(data_info["StorageDescriptor"]["Columns"])
    except KeyError:
        return "-"


def num_columns_fs(data_info):
    """Helper: Compute the number of columns from the feature group data"""
    try:
        return len(data_info["FeatureDefinitions"])
    except KeyError:
        return "-"


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
    from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp

    # Grab out SageMaker Session from the AWS Account Clamp
    sm_session = AWSAccountClamp().sagemaker_session()

    my_features = FeatureSetCore("test_features")
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)

    # Get the image URI with digest
    framework = "sklearn"
    region = "us-west-2"
    version = "1.2-1"
    # sm_session = SageSession()

    try:
        full_image_uri = get_image_uri_with_digest(framework, region, version, sm_session)
        print(f"Full Image URI with Digest: {full_image_uri}")
    except Exception as e:
        print(f"Error: {e}")

    # Test the newest files in an S3 folder method
    s3_path = "s3://sandbox-sageworks-artifacts/endpoints/inference/abalone-regression-end"
    most_recent = newest_files([s3_path], sm_session)

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
