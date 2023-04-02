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

    target = "class_number_of_rings"
    result_df[target] = result_df[target].astype(pd.Float64Dtype())
    rmse = sqrt(mean_squared_error(result_df[target], result_df["predictions"]))
    mae = mean_absolute_error(result_df[target], result_df["predictions"])
    r2 = r2_score(result_df[target], result_df["predictions"])
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")


if __name__ == "__main__":
    test()
