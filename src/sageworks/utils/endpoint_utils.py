"""Endpoint Utilities for SageWorks endpoints"""

import logging

import pandas as pd

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint

# Setup the logger
log = logging.getLogger("sageworks")


def predictions_using_fs(end: Endpoint) -> pd.DataFrame:
    """Code to run predictions using the FeatureSet

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (evaluation data)

    Returns:
        pd.DataFrame: Dataframe with the predictions using the FeatureSet data
    """
    # Grab the FeatureSet evaluation data
    feature_df = fs_evaluation_data(end)
    return end.predict(feature_df)


def fs_training_data(end: Endpoint) -> pd.DataFrame:
    """Code to get the training data from the FeatureSet used to train the Model

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (training data)

    Returns:
        pd.DataFrame: Dataframe with the features from the FeatureSet
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    fs = backtrack_to_fs(end)
    table = fs.get_training_view_table()
    train_df = fs.query(f"SELECT * FROM {table} where training = 1")
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
    table = fs.get_training_view_table()
    train_df = fs.query(f"SELECT * FROM {table} where training = 0")
    return train_df


def backtrack_to_fs(end: Endpoint) -> FeatureSet:
    """Code to Backtrack to FeatureSet: End -> Model -> FeatureSet

    Returns:
        FeatureSet: The FeatureSet used to train the Model
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    model_name = end.get_input()
    feature_name = Model(model_name).get_input()
    fs = FeatureSet(feature_name)
    return fs


if __name__ == "__main__":
    """Exercise the Endpoint Utilities"""

    # Create an Endpoint
    endpoint_name = "abalone-regression-end"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)

    # Get the training data
    train_df = fs_training_data(my_endpoint)
    print(train_df)

    # Make predictions on the Endpoint
    pred_output_df = predictions_using_fs(my_endpoint)
    print(pred_output_df)

    # Create a Classification Endpoint
    endpoint_name = "wine-classification-end"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)

    # Make predictions on the Endpoint
    pred_output_df = predictions_using_fs(my_endpoint)
    print(pred_output_df)
