"""Script that checks ModelGroups, Model (Resources), Endpoints and does a set of Sanity checks"""
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint

# Assuming AWSAccountClamp().sagemaker_client() gives you a SageMaker client
sagemaker_client = AWSAccountClamp().sagemaker_client()

# Setup logging
log = logging.getLogger("sageworks")


def ensure_health_tags():
    # Get all the model groups
    response = sagemaker_client.list_model_package_groups(MaxResults=100)
    model_group_names = [
        model_group["ModelPackageGroupName"] for model_group in response["ModelPackageGroupSummaryList"]
    ]
    log.important(f"Found {len(model_group_names)} Model Groups")
    # For each model group ensure the health tag storage is present
    for model_group_name in model_group_names:
        m = Model(model_group_name)
        m.sageworks_health_tags()
        m.details(recompute=True)

    # For each endpoint ensure the health tag storage is present
    endpoints_response = sagemaker_client.list_endpoints(MaxResults=100)
    for endpoint in endpoints_response["Endpoints"]:
        e = Endpoint(endpoint["EndpointName"])
        e.sageworks_health_tags()
        e.details(recompute=True)


if __name__ == "__main__":
    import os
    import argparse

    # We're going to have a --verbose flag and that's it
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Ensure that the REDIS_HOST ENV variable is set
    if not os.getenv("REDIS_HOST"):
        raise ValueError("REDIS_HOST ENV variable must be set")

    # Call the sanity checks
    ensure_health_tags()
