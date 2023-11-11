"""Script that checks ModelGroups, Model (Resources), Endpoints and does a set of Sanity checks"""
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.sageworks_logging import IMPORTANT_LEVEL_NUM

# Assuming AWSAccountClamp().sagemaker_client() gives you a SageMaker client
sagemaker_client = AWSAccountClamp().sagemaker_client()

# Setup logging
log = logging.getLogger("sageworks")


def run_sanity_checks(verbose: bool = False):

    # Set the log level based on the verbose flag
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(IMPORTANT_LEVEL_NUM)

    # Get all the model groups
    response = sagemaker_client.list_model_package_groups(MaxResults=100)
    model_group_names = [
        model_group["ModelPackageGroupName"]
        for model_group in response["ModelPackageGroupSummaryList"]
    ]
    log.important(f"Found {len(model_group_names)} Model Groups")
    for model_group_name in model_group_names:

        # For each model group report the number of model packages in the group
        package_response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_group_name
        )
        num_packages = len(package_response["ModelPackageSummaryList"])
        log.debug(f"Model Group: {model_group_name} ({num_packages} packages)")

    # Get all the model resources (models not in a model group)
    response = sagemaker_client.list_models(MaxResults=100)
    model_names = [model["ModelName"] for model in response["Models"]]
    log.important(f"Found {len(model_names)} Models (resources)")
    for model_name in model_names:
        log.debug(f"Model: {model_name}")

    # Each ModelGroup should have an endpoint and having an endpoint means
    # that a 'Model' Resource is created, so if we find a ModelGroup without
    # a Model Resource, then we have an ModelGroup without an endpoint
    model_group_set = set(model_group_names)
    model_set = set(model_names)
    model_group_without_model = model_group_set - model_set
    log.important(
        f"({len(model_group_without_model)}) Model Groups without an Endpoint: "
    )
    for model_group in model_group_without_model:
        log.important(f"\t{model_group}")
    log.important(
        "Recommendation: Delete these Models Groups or create an Endpoint for them"
    )

    # List all endpoints
    endpoints_response = sagemaker_client.list_endpoints(MaxResults=100)

    # Check each endpoint to see if it uses any of the models from the model group
    log.important(f"Found {len(endpoints_response['Endpoints'])} Endpoints")
    for endpoint in endpoints_response["Endpoints"]:
        log.info(f"Endpoint: {endpoint['EndpointName']}")
        endpoint_desc = sagemaker_client.describe_endpoint(
            EndpointName=endpoint["EndpointName"]
        )
        endpoint_config_desc = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )
        for variant in endpoint_config_desc["ProductionVariants"]:
            if variant["ModelName"] in model_names:
                log.debug(f"\t{endpoint['EndpointName']} --> {variant['ModelName']}")
            else:
                log.warning(
                    f"\t{endpoint['EndpointName']} --> Model not found: {variant['ModelName']}"
                )
                log.warning(
                    "Recommendation: This endpoint may no longer work, test it!"
                )


if __name__ == "__main__":
    import argparse

    # We're going to have a --verbose flag and that's it
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args, commands = parser.parse_known_args()
    verbose_arg = args.verbose

    # Call the sanity checks
    run_sanity_checks(verbose=verbose_arg)
