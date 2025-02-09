"""Experimental: Outlier detection based on nearest neighbors."""

from typing import Union
import pandas as pd


# Stub for the NearestNeighbors class
def get_neighbors(self, query_id: Union[int, str], include_self: bool = False):
    """
    Return neighbors of the given query ID, either by fixed neighbors or above a similarity threshold.

    Args:
        query_id (Union[int, str]): The ID of the query point.
        include_self (bool): Whether to include the query ID itself in the neighbor results.
    """
    return []


def compute_outliers(df: pd.DataFrame, target: str, id_column: str, model_type="classifier"):
    """
    Identifies outliers based on the target values of nearest neighbors.

    Args:
        model_type (str): Type of model to use for outlier detection. Defaults to "classifier".

    - For classification, an outlier is a point whose target differs from most of its neighbors.
    - For regression, an outlier is a point more than 3 standard deviations away from the mean of its neighbors.
    """
    if target is None:
        raise ValueError("Target column must be set to compute outliers.")

    outliers = []
    for idx in df.index:
        neighbors = get_neighbors(df.at[idx, id_column], include_self=False)

        if model_type == "classifier":
            majority_class = neighbors[target].mode()[0]
            is_outlier = df.at[idx, target] != majority_class

        else:  # Regression
            neighbor_mean = neighbors[target].mean()
            neighbor_std = neighbors[target].std()
            is_outlier = abs(df.at[idx, target] - neighbor_mean) > 3 * neighbor_std

        outliers.append(is_outlier)

    # Add the outlier column to the DataFrame
    df["outlier"] = outliers
