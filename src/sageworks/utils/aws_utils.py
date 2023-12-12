import botocore
import logging
import time
import json
import base64
from sagemaker.session import Session as SageSession

# SageWorks Imports
from sageworks.utils.trace_calls import trace_calls

# SageWorks Logger
log = logging.getLogger("sageworks")


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


@trace_calls
def sagemaker_retrieve_tags(arn: str, sm_session: SageSession) -> dict:
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
                log.warning(f"ThrottlingException: list_tags on {arn}")
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                # Handle other ClientErrors that may occur
                log.error(f"Unexpected ClientError: {error_code}")
                raise e


def _aws_tags_to_dict(aws_tags) -> dict:
    """Internal: AWS Tags are in an odd format, so convert to regular dictionary"""

    def decode_value(value):
        try:
            return json.loads(base64.b64decode(value).decode("utf-8"))
        except Exception:
            return value

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
        stitched_json_str = base64.b64decode(stitched_base64_str).decode("utf-8")
        stitched_dict = json.loads(stitched_json_str)

        regular_tags[base_key] = stitched_dict

    return regular_tags


def _chunk_dict_to_aws_tags(base_key: str, data: dict) -> dict:
    # Internal: Convert dictionary to minified JSON string
    json_str = json.dumps(data, separators=(",", ":"))

    # Encode JSON string to base64
    base64_str = base64.b64encode(json_str.encode()).decode()

    # Initialize variables
    chunk_size = 256  # Max size for AWS tag value
    chunks = {}

    # Split base64 string into chunks and create tags
    for i in range(0, len(base64_str), chunk_size):
        chunk = base64_str[i : i + chunk_size]
        chunks[f"{base_key}_chunk_{i // chunk_size + 1}"] = chunk

    return chunks


if __name__ == "__main__":
    """Exercise the AWS Utils"""
    from pprint import pprint
    from sageworks.artifacts.feature_sets.feature_set import FeatureSet

    my_features = FeatureSet("test_feature_set")
    my_meta = my_features.sageworks_meta()
    pprint(my_meta)
