# This code shows how workbench stores and retrieves tags from AWS
# You could use this code to retrieve tags from AWS and display them in a UI

import sagemaker
import base64
import json
from pprint import pprint

model_group_name = "abalone-regression"


# Helper method to decode tags
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


# Helper method to stitch together tags
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


sm_session = sagemaker.Session()
client = sm_session.sagemaker_client

# Retrieve the model package group ARN
response = client.describe_model_package_group(ModelPackageGroupName=model_group_name)
group_arn = response["ModelPackageGroupArn"]

# Use the SageMaker session's list_tags method
tags = sm_session.list_tags(resource_arn=group_arn)

# Convert tags to dictionary
tags = aws_tags_to_dict(tags)
pprint(tags)
