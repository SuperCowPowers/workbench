"""Endpoint Utilities for SageWorks endpoints"""

import logging
from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint

# Setup the logger
log = logging.getLogger("sageworks")


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
    endpoint_name = "abalone-regression-end"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)

    # Get the training data
    my_train_df = fs_training_data(my_endpoint)
    print(my_train_df)

    # Get the evaluation data
    my_eval_df = fs_evaluation_data(my_endpoint)
    print(my_eval_df)

    # Backtrack to the FeatureSet
    my_fs = backtrack_to_fs(my_endpoint)
    print(my_fs)
