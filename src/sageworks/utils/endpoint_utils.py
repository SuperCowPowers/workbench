"""Endpoint Utilities for SageWorks endpoints"""

import logging

import pandas as pd

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint

# Setup the logger
log = logging.getLogger("sageworks")


def auto_capture_metrics(end: Endpoint) -> None:
    """Code to Auto Capture Performance Metrics for an Endpoint

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (evaluation data)
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    feature_df = fs_evaluation_data(end)
    model_details = Model(end.get_input()).details()
    target_column = model_details.get("sageworks_model_target")
    if target_column is None:
        log.warning("No target column for the model, aborting Auto Capture Metrics...")
        return
    end.capture_performance_metrics(feature_df, target_column, "auto", "auto", "Auto Captured Metrics")


def fs_predictions(end: Endpoint) -> pd.DataFrame:
    """Code to get the predictions from the FeatureSet

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (evaluation data)

    Returns:
        pd.DataFrame: Dataframe with the predictions from the FeatureSet
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


def backtrack_to_fs(end: Endpoint) -> pd.DataFrame:
    """Code to Backtrack to FeatureSet: End -> Model -> FeatureSet

    Returns:
        pd.DataFrame: The training data in a dataframe with the features from the FeatureSet
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

    # Make predictions on the Endpoint
    pred_output_df = fs_predictions(my_endpoint)
    print(pred_output_df)

    # Capture performance metrics
    auto_capture_metrics(my_endpoint)
