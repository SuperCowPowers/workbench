# This code shows how workbench stores and retrieves tags from AWS
# You could use this code to retrieve tags from AWS and display them in a UI

import sagemaker
from pprint import pprint

from workbench.utils.aws_utils import aws_tags_to_dict

model_group_name = "abalone-regression"

sm_session = sagemaker.Session()
client = sm_session.sagemaker_client

# Retrieve the model package group ARN
response = client.describe_model_package_group(ModelPackageGroupName=model_group_name)
group_arn = response["ModelPackageGroupArn"]

# Use the SageMaker session's list_tags method
tags = sm_session.list_tags(resource_arn=group_arn)

# Convert tags to a dictionary (handles the b64: marker, chunking, and decoding)
tags = aws_tags_to_dict(tags)
pprint(tags)
