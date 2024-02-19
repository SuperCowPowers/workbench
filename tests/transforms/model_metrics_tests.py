"""Tests for the creation and comparison of Model Metrics"""

from pprint import pprint

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint


# Test the Model Metrics
def test_metrics():
    """Test the Model Metrics"""
    my_model = Model("abalone-regression")
    metrics = my_model.model_metrics()
    pprint(metrics)


def test_auto_inference():
    # Run auto_inference (back track to FeatureSet)
    my_endpoint = Endpoint("abalone-regression-end")
    pred_results = my_endpoint.auto_inference()
    pprint(pred_results.head())


def test_inference():
    # Run inference on the model

    # Grab a dataframe for inference
    my_features = FeatureSet("abalone_features")
    table = my_features.get_training_view_table()
    df = my_features.query(f"SELECT * FROM {table} where training = 0")

    # Run inference
    my_endpoint = Endpoint("abalone-regression-end")
    pred_results = my_endpoint.inference(df)
    pprint(pred_results.head())


def create_model_and_endpoint():
    # Retrieve an existing FeatureSet
    my_features = FeatureSet("abalone_features")

    # Create a Model/Endpoint from the FeatureSet
    my_model = my_features.to_model(model_type=ModelType.REGRESSOR, target_column="class_number_of_rings")
    my_model.to_endpoint(name="abalone-regression-end", tags=["abalone", "public"])


if __name__ == "__main__":

    # Set Pandas display options
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    # Create/Recreate the Model and Endpoint
    # create_model_and_endpoint()

    # Run the tests
    test_metrics()
    test_auto_inference()
    test_inference()
