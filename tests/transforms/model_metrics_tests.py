"""Tests for the creation and comparison of Model Metrics"""

from pprint import pprint

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint


model_reg = Model("abalone-regression")
model_class = Model("wine-classification")


# Test the Model Metrics
def test_metrics():
    """Test the Model Metrics"""
    pprint(model_reg.inference_metadata())
    pprint(model_reg.model_metrics())
    pprint(model_class.model_metrics())


def test_validation_predictions():
    pprint(model_reg.validation_predictions())
    pprint(model_class.validation_predictions())


def test_inference_predictions():
    pprint(model_reg.inference_predictions())
    pprint(model_class.inference_predictions())


def test_confusion_matrix():
    pprint(model_reg.confusion_matrix())
    pprint(model_class.confusion_matrix())


def test_shap_values():
    pprint(model_reg.shapley_values())
    pprint(model_class.shapley_values())


def test_metrics_with_capture_uuid():
    """Test the Model Metrics"""
    metrics = model_reg.model_metrics("featureset_20")
    pprint(metrics)
    metrics = model_class.model_metrics("featureset_20")
    pprint(metrics)


def test_auto_inference():
    # Run auto_inference (back track to FeatureSet)
    my_endpoint = Endpoint("abalone-regression-end")
    pred_results = my_endpoint.auto_inference()
    pprint(pred_results.head())

    my_endpoint = Endpoint("wine-classification-end")
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
    model_reg = my_features.to_model(model_type=ModelType.REGRESSOR, target_column="class_number_of_rings")
    model_reg.to_endpoint(name="abalone-regression-end", tags=["abalone", "public"])


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
    test_validation_predictions()
    test_inference_predictions()
    test_shap_values()
    test_auto_inference()
    test_inference()
