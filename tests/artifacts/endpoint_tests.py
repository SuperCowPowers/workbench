"""Tests for the Endpoint functionality"""
import pandas as pd

# SageWorks Imports
from sageworks.artifacts.endpoints.endpoint import Endpoint


def test():
    """Simple test of the Endpoint functionality"""
    from sageworks.transforms.pandas_transforms.features_to_pandas import (
        FeaturesToPandas,
    )
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from math import sqrt

    # Grab an Endpoint object and pull some information from it
    my_endpoint = Endpoint("abalone-regression-endpoint")

    # Call the various methods

    # Let's do a check/validation of the Endpoint
    assert my_endpoint.check()

    # Creation/Modification Times
    print(my_endpoint.created())
    print(my_endpoint.modified())

    # Get the tags associated with this Endpoint
    print(f"Tags: {my_endpoint.sageworks_tags()}")

    # Create the FeatureSet to DF Transform
    feature_to_pandas = FeaturesToPandas("abalone_feature_set")

    # Transform the DataSource into a Pandas DataFrame (with max_rows = 100)
    feature_to_pandas.transform(max_rows=100)

    # Grab the output and show it
    feature_df = feature_to_pandas.get_output()
    print(feature_df)

    # Okay now run inference against our Features DataFrame
    result_df = my_endpoint.predict(feature_df)
    print(result_df)

    # Compute performance metrics for our test predictions
    target_column = "class_number_of_rings"
    metrics = my_endpoint.performance_metrics(target_column, prediction_df)
    for metric, value in metrics.items():
        print(f"{metric}: {value:0.3f}")


if __name__ == "__main__":
    test()
