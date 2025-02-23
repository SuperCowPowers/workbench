"""Endpoint Utilities for Workbench endpoints"""

import boto3
import logging
from typing import Union, Optional
import pandas as pd

# Workbench Imports
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model
from workbench.api.endpoint import Endpoint

# Set up the log
log = logging.getLogger("workbench")


def internal_model_data_url(endpoint_config_name: str, session: boto3.Session) -> Optional[str]:
    """
    Retrieves the S3 URL of the model.tar.gz file associated with a SageMaker endpoint configuration.

    Args:
        endpoint_config_name (str): The name of the SageMaker endpoint configuration.
        session (boto3.Session): An active boto3 session.

    Returns:
        Optional[str]: S3 URL of the model.tar.gz file if found, otherwise None.
    """
    try:
        sagemaker_client = session.client("sagemaker")

        # Retrieve the Endpoint Config
        endpoint_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)

        # Extract Model Name from Production Variants
        production_variants = endpoint_config.get("ProductionVariants", [])
        if not production_variants:
            log.critical(f"No production variants found for endpoint config: {endpoint_config_name}")
            return None

        model_name = production_variants[0].get("ModelName")
        if not model_name:
            log.critical(f"No model name found in production variants for endpoint config: {endpoint_config_name}")
            return None

        # Retrieve Model Details
        model_details = sagemaker_client.describe_model(ModelName=model_name)
        containers = model_details.get("Containers")
        if containers:
            # Handle serverless or multi-container models
            model_package_name = containers[0].get("ModelPackageName")
            if model_package_name:
                log.info(f"Model package name found: {model_package_name}")

                # Describe the model package to get the ModelDataUrl
                model_package_details = sagemaker_client.describe_model_package(ModelPackageName=model_package_name)
                model_data_url = (
                    model_package_details.get("InferenceSpecification", {})
                    .get("Containers", [{}])[0]
                    .get("ModelDataUrl")
                )
                if model_data_url:
                    log.info(f"Model data URL from package: {model_data_url}")
                    return model_data_url

        # Handle standard models
        model_data_url = model_details.get("PrimaryContainer", {}).get("ModelDataUrl")
        if model_data_url:
            log.info(f"Model data URL found: {model_data_url}")
            return model_data_url

        log.critical(f"No model data or package details found for model: {model_name}")
        return None

    except Exception as e:
        log.critical(f"Error retrieving model data URL for endpoint config {endpoint_config_name}: {e}")
        return None


def fs_training_data(end: Endpoint) -> pd.DataFrame:
    """Code to get the training data from the FeatureSet used to train the Model

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (training data)

    Returns:
        pd.DataFrame: Dataframe with the features from the FeatureSet
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    fs = backtrack_to_fs(end)

    # Sanity check that we have a FeatureSet
    if fs is None:
        log.error("No FeatureSet found for this endpoint. Returning empty dataframe.")
        return pd.DataFrame()

    # Get the training data
    table = fs.view("training").table
    train_df = fs.query(f'SELECT * FROM "{table}" where training = TRUE')
    return train_df


def fs_evaluation_data(end: Endpoint) -> pd.DataFrame:
    """Code to get the evaluation data from the FeatureSet NOT used for training

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (evaluation data)

    Returns:
        pd.DataFrame: The training data in a dataframe with the features from the FeatureSet
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    fs = backtrack_to_fs(end)

    # Sanity check that we have a FeatureSet
    if fs is None:
        log.error("No FeatureSet found for this endpoint. Returning empty dataframe.")
        return pd.DataFrame()

    # Get the evaluation data
    table = fs.view("training").table
    eval_df = fs.query(f'SELECT * FROM "{table}" where training = FALSE')
    return eval_df


def backtrack_to_fs(end: Endpoint) -> Union[FeatureSet, None]:
    """Code to Backtrack to FeatureSet: End -> Model -> FeatureSet

    Returns:
        FeatureSet (Union[FeatureSet, None]): The FeatureSet object or None if not found
    """

    # Sanity Check that we have a model
    model = Model(end.get_input())
    if not model.exists():
        log.error("No model found for this endpoint. Returning None.")
        return None

    # Now get the FeatureSet and make sure it exists
    fs = FeatureSet(model.get_input())
    if not fs.exists():
        log.error("No FeatureSet found for this endpoint. Returning None.")
        return None

    # Return the FeatureSet
    return fs


if __name__ == "__main__":
    """Exercise the Endpoint Utilities"""

    # Create an Endpoint
    endpoint_name = "abalone-regression"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)

    # Get the Model Data URL
    model_data_url = internal_model_data_url(my_endpoint.endpoint_config_name(), my_endpoint.boto3_session)
    print(model_data_url)

    # Get the training data
    my_train_df = fs_training_data(my_endpoint)
    print(my_train_df)

    # Get the evaluation data
    my_eval_df = fs_evaluation_data(my_endpoint)
    print(my_eval_df)

    # Backtrack to the FeatureSet
    my_fs = backtrack_to_fs(my_endpoint)
    print(my_fs)

    # Also test for realtime endpoints
    rt_endpoint = Endpoint("abalone-regression-end-rt")
    if rt_endpoint.exists():
        model_data_url = internal_model_data_url(rt_endpoint.endpoint_config_name(), rt_endpoint.boto3_session)
        print(model_data_url)
