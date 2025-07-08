"""Tests for the creation and comparison of Model Metrics"""

import pytest
from pprint import pprint

# Workbench Imports
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.api.endpoint import Endpoint


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


def test_retrieval_with_capture_name():
    """Test the retrieval of the model metrics using capture Name"""
    capture_list = model_class.list_inference_runs()
    for capture_name in capture_list:
        print(f"\n\n*** Retrieval with Capture Name ({capture_name}) ***")
        pprint(model_class.get_inference_metadata(capture_name).head())
        pprint(model_class.get_inference_metrics(capture_name).head())
        pprint(model_class.get_inference_predictions(capture_name).head())
        pprint(model_class.confusion_matrix(capture_name))


def test_validation_predictions():
    print("\n\n*** Validation Predictions ***")
    reg_val_preds = model_reg._get_validation_predictions()
    pprint(reg_val_preds.head())
    class_val_preds = model_class._get_validation_predictions()
    pprint(class_val_preds.head())


def test_inference_predictions():
    print("\n\n*** Inference Predictions ***")

    # Make sure we have inference predictions
    end = Endpoint("abalone-regression")
    end.auto_inference(capture=True)
    end = Endpoint("wine-classification")
    end.auto_inference(capture=True)

    # Retrieve the inference predictions
    model_reg = Model("abalone-regression")
    if model_reg.get_inference_predictions() is None:
        print(f"Model {model_reg.name} has no inference predictions!")
        exit(1)
    pprint(model_reg.get_inference_predictions().head())
    if model_class.get_inference_predictions() is None:
        print(f"Model {model_class.name} has no inference predictions!")
        exit(1)
    pprint(model_class.get_inference_predictions().head())


def test_confusion_matrix():
    print("\n\n*** Confusion Matrix ***")
    pprint(model_reg.confusion_matrix())
    pprint(model_class.confusion_matrix())


def test_shap_values():
    print("\n\n*** SHAP Features (regression) ***")
    shap_features = model_reg.shap_importance()
    if shap_features is None:
        print(f"Model {model_reg.name} has no SHAP features!")
    else:
        pprint(shap_features)

    print("\n\n*** SHAP Data (regression) ***")
    shap_data = model_reg.shap_data()
    if shap_data is None:
        print(f"Model {model_reg.name} has no SHAP data!")
    else:
        pprint(shap_features)

    print("\n\n*** SHAP Features (classification) ***")
    shap_features = model_class.shap_importance()
    if shap_features is None:
        print(f"Model {model_class.name} has no SHAP features!")
    else:
        pprint(shap_features)

    print("\n\n*** SHAP Data (classification) ***")
    shap_data = model_class.shap_data()
    if shap_data is None:
        print(f"Model {model_class.name} has no SHAP data!")
    else:
        # Classifiers have a dictionary of dataframes
        for key, df in shap_data.items():
            print(f"{key}")
            print(df.head())


def test_metrics_with_capture_name():
    """Test the Performance Metrics using a Capture Name"""
    metrics = model_reg.get_inference_metrics("auto_inference")
    print("\n\n*** Performance Metrics with Capture Name ***")
    pprint(metrics)
    metrics = model_class.get_inference_metrics("auto_inference")
    pprint(metrics)


@pytest.mark.long
def test_auto_inference():
    # Run auto_inference (back track to FeatureSet)
    my_endpoint = Endpoint("abalone-regression")
    pred_results = my_endpoint.auto_inference()
    print("\n\n*** Auto Inference ***")
    pprint(pred_results.head())

    my_endpoint = Endpoint("wine-classification")
    pred_results = my_endpoint.auto_inference()
    pprint(pred_results.head())


@pytest.mark.long
def test_inference_with_capture_name():
    # Run inference on the model
    capture_name = "my_holdout_test"

    # Grab a dataframe for inference
    my_features = FeatureSet("abalone_features")
    table = my_features.view("training").table
    df = my_features.query(f'SELECT * FROM "{table}" where training = FALSE')

    # Run inference
    my_endpoint = Endpoint("abalone-regression")
    pred_results = my_endpoint.inference(df, capture_name)
    pprint(pred_results.head())


def create_model_and_endpoint():
    # Retrieve an existing FeatureSet
    my_features = FeatureSet("abalone_features")

    # Create a Model/Endpoint from the FeatureSet
    model_reg = my_features.to_model(
        name="abalone-regression", model_type=ModelType.REGRESSOR, target_column="class_number_of_rings"
    )
    model_reg.to_endpoint(name="abalone-regression", tags=["abalone", "public"])


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
    test_retrieval_with_capture_name()
    test_validation_predictions()
    test_inference_predictions()
    test_confusion_matrix()
    test_shap_values()
    test_metrics_with_capture_name()

    # These are longer tests (commented out for now)
    # test_auto_inference()
    # test_inference_with_capture_name("my_holdout_test")
