"""Tests for the Endpoint functionality"""

# SageWorks Imports
from sageworks.core.artifacts.endpoint_core import EndpointCore


def test():
    """Simple test of the Endpoint functionality"""
    from sageworks.core.transforms.pandas_transforms.features_to_pandas import FeaturesToPandas

    # Grab an Endpoint object and pull some information from it
    my_endpoint = EndpointCore("abalone-regression-end")

    # Call the various methods

    # Let's do a check/validation of the Endpoint
    assert my_endpoint.exists()

    # Creation/Modification Times
    print(my_endpoint.created())
    print(my_endpoint.modified())

    # Get the tags associated with this Endpoint
    print(f"Tags: {my_endpoint.get_tags()}")

    # Create the FeatureSet to DF Transform
    feature_to_pandas = FeaturesToPandas("abalone_features")

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
    metrics = my_endpoint.regression_metrics(target_column, result_df)
    print(metrics)


if __name__ == "__main__":
    test()
