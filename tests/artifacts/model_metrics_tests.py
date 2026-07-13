"""Tests for the creation and comparison of Model Metrics"""

import pytest
from pprint import pprint

# Workbench Imports
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType, ModelFramework
from workbench.api.endpoint import Endpoint


@pytest.fixture(scope="module")
def model_reg():
    return Model("abalone-regression")


@pytest.fixture(scope="module")
def model_class():
    return Model("wine-classification")


# Test the Model Metrics
def test_list_inference_runs(model_reg, model_class):
    """Test the List Inference Runs"""
    print("\n\n*** List Inference Runs ***")
    pprint(model_reg.list_inference_runs())
    pprint(model_class.list_inference_runs())


def test_performance_metrics(model_reg, model_class):
    """Test the Model Performance Metrics"""
    print("\n\n*** Performance Metrics ***")
    pprint(model_reg.get_inference_metadata())
    pprint(model_reg.get_inference_metrics())
    pprint(model_class.get_inference_metrics())


def test_retrieval_with_capture_name(model_class):
    """Test the retrieval of the model metrics using capture Name"""
    capture_list = model_class.list_inference_runs()
    for capture_name in capture_list:
        print(f"\n\n*** Retrieval with Capture Name ({capture_name}) ***")
        pprint(model_class.get_inference_metadata(capture_name).head())
        pprint(model_class.get_inference_metrics(capture_name).head())
        pprint(model_class.get_inference_predictions(capture_name).head())
        pprint(model_class.confusion_matrix(capture_name))


def test_validation_predictions(model_reg, model_class):
    print("\n\n*** Validation Predictions ***")
    # "model_training" routes to the validation predictions through the public API
    reg_val_preds = model_reg.get_inference_predictions("model_training")
    assert reg_val_preds is not None, f"{model_reg.name} has no validation predictions"
    pprint(reg_val_preds.head())
    class_val_preds = model_class.get_inference_predictions("model_training")
    assert class_val_preds is not None, f"{model_class.name} has no validation predictions"
    pprint(class_val_preds.head())


@pytest.mark.medium
def test_inference_predictions(model_class):
    print("\n\n*** Inference Predictions ***")

    # Make sure we have inference predictions
    end = Endpoint("abalone-regression")
    end.test_inference()
    end = Endpoint("wine-classification")
    end.test_inference()

    # Retrieve the inference predictions
    model_reg = Model("abalone-regression")
    if model_reg.get_inference_predictions() is None:
        pytest.fail(f"Model {model_reg.name} has no inference predictions!")
    pprint(model_reg.get_inference_predictions().head())
    if model_class.get_inference_predictions() is None:
        pytest.fail(f"Model {model_class.name} has no inference predictions!")
    pprint(model_class.get_inference_predictions().head())


def test_confusion_matrix(model_reg, model_class):
    print("\n\n*** Confusion Matrix ***")
    pprint(model_reg.confusion_matrix())
    pprint(model_class.confusion_matrix())


def test_shap_values(model_reg, model_class):
    print("\n\n*** SHAP Features (regression) ***")
    shap_features = model_reg.shap_importance()
    if shap_features is None:
        print(f"Model {model_reg.name} has no SHAP features!")
    else:
        pprint(shap_features)

    print("\n\n*** SHAP Values (regression) ***")
    shap_vals = model_reg.shap_values()
    if shap_vals is None:
        print(f"Model {model_reg.name} has no SHAP values!")
    else:
        print(shap_vals.head())

    print("\n\n*** SHAP Features (classification) ***")
    shap_features = model_class.shap_importance()
    if shap_features is None:
        print(f"Model {model_class.name} has no SHAP features!")
    else:
        pprint(shap_features)

    print("\n\n*** SHAP Values (classification) ***")
    shap_vals = model_class.shap_values()
    if shap_vals is None:
        print(f"Model {model_class.name} has no SHAP values!")
    else:
        # Classifiers have a dictionary of dataframes
        for key, df in shap_vals.items():
            print(f"{key}")
            print(df.head())


def test_metrics_with_capture_name(model_reg, model_class):
    """Test the Performance Metrics using a Capture Name"""
    metrics = model_reg.get_inference_metrics("test_inference")
    print("\n\n*** Performance Metrics with Capture Name ***")
    pprint(metrics)
    metrics = model_class.get_inference_metrics("test_inference")
    pprint(metrics)


@pytest.mark.long
def test_endpoint_inference():
    # Run test_inference (back track to FeatureSet)
    my_endpoint = Endpoint("abalone-regression")
    pred_results = my_endpoint.test_inference()
    print("\n\n*** Test Inference ***")
    pprint(pred_results.head())

    my_endpoint = Endpoint("wine-classification")
    pred_results = my_endpoint.test_inference()
    pprint(pred_results.head())


@pytest.mark.long
def test_inference_with_capture_name():
    # Run inference on the model
    capture_name = "my_holdout_test"

    # Grab a dataframe for inference
    my_features = FeatureSet("abalone_features")
    df = my_features.pull_dataframe()

    # Run inference
    my_endpoint = Endpoint("abalone-regression")
    pred_results = my_endpoint.inference(df, capture_name)
    pprint(pred_results.head())


def create_model_and_endpoint():
    # Retrieve an existing FeatureSet
    my_features = FeatureSet("abalone_features")

    # Create a Model/Endpoint from the FeatureSet
    model_reg = my_features.to_model(
        name="abalone-regression",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="class_number_of_rings",
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

    # Construct the models directly (pytest fixtures provide these during test runs)
    reg_model = Model("abalone-regression")
    class_model = Model("wine-classification")

    # Run the tests
    test_list_inference_runs(reg_model, class_model)
    test_performance_metrics(reg_model, class_model)
    test_retrieval_with_capture_name(class_model)
    test_validation_predictions(reg_model, class_model)
    test_inference_predictions(class_model)
    test_confusion_matrix(reg_model, class_model)
    test_shap_values(reg_model, class_model)
    test_metrics_with_capture_name(reg_model, class_model)

    # These are longer tests (commented out for now)
    # test_endpoint_inference()
    # test_inference_with_capture_name("my_holdout_test")
