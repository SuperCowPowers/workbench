"""Tests for the creation and comparison of Model Metrics"""

import pytest
from pprint import pprint

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint


model_reg = Model("abalone-regression")
model_class = Model("wine-classification")


# Test the Model Metrics
def test_list_inference_runs():
    """Test the List Inference Runs"""
    print("\n\n*** List Inference Runs ***")
    pprint(model_reg.list_inference_runs())
    pprint(model_class.list_inference_runs())


def test_performance_metrics():
    """Test the Model Performance Metrics"""
    print("\n\n*** Performance Metrics ***")
    pprint(model_reg.get_inference_metadata())
    pprint(model_reg.get_inference_metrics())
    pprint(model_class.get_inference_metrics())


def test_retrieval_with_capture_uuid():
    """Test the retrieval of the model metrics using capture UUID"""
    capture_list = model_class.list_inference_runs()
    for capture_uuid in capture_list:
        print(f"\n\n*** Retrieval with Capture UUID ({capture_uuid}) ***")
        pprint(model_class.get_inference_metadata(capture_uuid).head())
        pprint(model_class.get_inference_metrics(capture_uuid).head())
        pprint(model_class.get_inference_predictions(capture_uuid).head())
        pprint(model_class.confusion_matrix(capture_uuid))
        # Classifiers have a list of dataframes for shap values
        shap_list = model_class.shapley_values(capture_uuid)
        for i, df in enumerate(shap_list):
            print(f"SHAP Values for Class {i}")
            pprint(df.head())


def test_validation_predictions():
    print("\n\n*** Validation Predictions ***")
    pprint(model_reg._get_validation_predictions().head())
    pprint(model_class._get_validation_predictions().head())


def test_inference_predictions():
    print("\n\n*** Inference Predictions ***")
    if model_reg.get_inference_predictions() is None:
        print(f"Model {model_reg.uuid} has no inference predictions!")
        exit(1)
    pprint(model_reg.get_inference_predictions().head())
    if model_class.get_inference_predictions() is None:
        print(f"Model {model_class.uuid} has no inference predictions!")
        exit(1)
    pprint(model_class.get_inference_predictions().head())


def test_confusion_matrix():
    print("\n\n*** Confusion Matrix ***")
    pprint(model_reg.confusion_matrix())
    pprint(model_class.confusion_matrix())


def test_shap_values():
    print("\n\n*** SHAP Values ***")
    pprint(model_reg.shapley_values().head())

    # Classifiers have a list of dataframes
    shap_list = model_class.shapley_values()
    for i, df in enumerate(shap_list):
        print(f"SHAP Values for Class {i}")
        pprint(df.head())


def test_metrics_with_capture_uuid():
    """Test the Performance Metrics using a Capture UUID"""
    metrics = model_reg.get_inference_metrics("training_holdout")
    print("\n\n*** Performance Metrics with Capture UUID ***")
    pprint(metrics)
    metrics = model_class.get_inference_metrics("training_holdout")
    pprint(metrics)


@pytest.mark.long
def test_auto_inference():
    # Run auto_inference (back track to FeatureSet)
    my_endpoint = Endpoint("abalone-regression-end")
    pred_results = my_endpoint.auto_inference()
    print("\n\n*** Auto Inference ***")
    pprint(pred_results.head())

    my_endpoint = Endpoint("wine-classification-end")
    pred_results = my_endpoint.auto_inference()
    pprint(pred_results.head())


@pytest.mark.long
def test_inference_with_capture_uuid():
    # Run inference on the model
    capture_uuid = "my_holdout_test"

    # Grab a dataframe for inference
    my_features = FeatureSet("abalone_features")
    table = my_features.get_training_view_table()
    df = my_features.query(f"SELECT * FROM {table} where training = 0")

    # Run inference
    my_endpoint = Endpoint("abalone-regression-end")
    pred_results = my_endpoint.inference(df, capture_uuid)
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
    test_list_inference_runs()
    test_performance_metrics()
    test_retrieval_with_capture_uuid()
    test_validation_predictions()
    test_inference_predictions()
    test_confusion_matrix()
    test_shap_values()
    test_metrics_with_capture_uuid()

    # These are longer tests (commented out for now)
    # test_auto_inference()
    # test_inference_with_capture_uuid("my_holdout_test")
