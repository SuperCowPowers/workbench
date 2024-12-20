"""Script that checks ModelGroups, Model (Resources), Endpoints and does a set of Sanity checks"""

import logging

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.core.artifacts.model_core import ModelCore
from workbench.core.artifacts.endpoint_core import EndpointCore

# Get a sagemaker_client from the AWSAccountClamp()
sagemaker_client = AWSAccountClamp().sagemaker_client()

# Setup logging
log = logging.getLogger("workbench")
log.setLevel(logging.INFO)


def run_sanity_checks(tag: bool = False, delete_stuff: bool = False):

    # Get all the model groups
    response = sagemaker_client.list_model_package_groups(MaxResults=100)
    model_group_names = [
        model_group["ModelPackageGroupName"] for model_group in response["ModelPackageGroupSummaryList"]
    ]
    log.important(f"Found {len(model_group_names)} Model Groups")
    for model_group_name in model_group_names:
        # For each model group report the number of model packages in the group
        package_response = sagemaker_client.list_model_packages(ModelPackageGroupName=model_group_name)
        num_packages = len(package_response["ModelPackageSummaryList"])
        log.info(f"Model Group: {model_group_name} ({num_packages} packages)")

    # Get all the model packages
    all_model_packages = sagemaker_client.list_model_packages(MaxResults=100)

    # Figure out with model packages are NOT part of a model package group
    standalone_model_packages = []
    for package in all_model_packages["ModelPackageSummaryList"]:
        if "ModelPackageGroupName" in package:
            if package["ModelPackageGroupName"] not in model_group_names:
                standalone_model_packages.append(package)
        else:
            standalone_model_packages.append(package)
    log.important(f"Found {len(standalone_model_packages)} Model Packages (not in a Group)")
    for package in standalone_model_packages:
        log.important(f"\t{package['ModelPackageArn']}")

    # Get all the model resources (models not in a model group)
    response = sagemaker_client.list_models(MaxResults=100)
    model_names = [model["ModelName"] for model in response["Models"]]
    log.important(f"Found {len(model_names)} Models (not in a Model Package/Group)")
    for model_name in model_names:
        log.info(f"\tModel: {model_name}")

    # Each ModelGroup should have an endpoint and having an endpoint means
    # that a 'Model' Resource is created, so if we find a ModelGroup without
    # a Model Resource, then we have an ModelGroup without an endpoint
    model_group_set = set(model_group_names)
    model_set = set(model_names)
    model_group_without_model = []

    # Check if the model group name is a substring of any model name
    for model_group in model_group_set:
        if not any(model_group in model_name for model_name in model_set):
            model_group_without_model.append(model_group)
    if len(model_group_without_model) > 0:
        log.important(
            f"({len(model_group_without_model)}) Possible Model Groups without an Endpoint (Heuristic/substring): "
        )
        for model_group in model_group_without_model:
            log.important(f"{model_group}")
            if tag:
                m = ModelCore(model_group)
                m.add_health_tag("no_endpoint")
        log.important("Recommendation: Delete these Models Groups or create an Endpoint for them")

    # List all endpoints
    endpoints_response = sagemaker_client.list_endpoints(MaxResults=100)

    # Check each endpoint to see if it uses any of the models from the model group
    found_models = []
    log.important(f"Found {len(endpoints_response['Endpoints'])} Endpoints")
    for endpoint in endpoints_response["Endpoints"]:
        log.info(f"Endpoint: {endpoint['EndpointName']}")
        endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint["EndpointName"])
        endpoint_config_desc = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )
        for variant in endpoint_config_desc["ProductionVariants"]:
            if variant["ModelName"] in model_names:
                log.info(f"\t{endpoint['EndpointName']} --> {variant['ModelName']}")
                found_models.append(variant["ModelName"])
            else:
                log.warning(f"\t{endpoint['EndpointName']} --> Model not found: {variant['ModelName']}")
                log.warning("Recommendation: This endpoint may no longer work, test it!")
                if tag:
                    e = EndpointCore(endpoint["EndpointName"])
                    e.add_health_tag("no_model")

    # Now report on the models that are not used by any endpoint (orphans)
    unused_models = model_set - set(found_models)
    log.important(f"Found {len(unused_models)} Model Resources without an Endpoint (Orphans) ")
    for model in unused_models:
        log.important(f"\t{model}")
    log.important("Recommendation: Delete these Models")
    if delete_stuff:
        for model in unused_models:
            delete_model(model, sagemaker_client)
            ModelCore.managed_delete(model)


def delete_model(model_name: str, sm_client):
    log.important(f"Deleting Model '{model_name}'...")
    sm_client.delete_model(ModelName=model_name)


if __name__ == "__main__":
    import argparse

    # We're going to have a --verbose flag and that's it
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="store_true")
    parser.add_argument("--delete-stuff", action="store_true")
    args, commands = parser.parse_known_args()

    # Call the sanity checks
    run_sanity_checks(tag=args.tag, delete_stuff=args.delete_stuff)
