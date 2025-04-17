"""Tests the support for Empty Model Groups"""

from pprint import pprint

# Workbench Imports
from workbench.api.model import Model


model = Model("empty-model-group")


# Test the Model Metrics
def test_list_inference_runs():
    """Test the List Inference Runs"""
    print("\n\n*** List Inference Runs ***")
    pprint(model.list_inference_runs())


def test_performance_metrics():
    """Test the Model Performance Metrics"""
    print("\n\n*** Performance Metrics ***")
    pprint(model.get_inference_metadata())


def test_retrieval_with_capture_uuid():
    """Test the retrieval of the model metrics using capture UUID"""
    capture_list = model.list_inference_runs()
    for capture_uuid in capture_list:
        print(f"\n\n*** Retrieval with Capture UUID ({capture_uuid}) ***")
        pprint(model.get_inference_metadata(capture_uuid).head())
        pprint(model.get_inference_metrics(capture_uuid).head())
        pprint(model.get_inference_predictions(capture_uuid).head())
        pprint(model.confusion_matrix(capture_uuid))


def test_validation_predictions():
    print("\n\n*** Validation Predictions ***")
    val_predictions = model._get_validation_predictions()
    if val_predictions is None:
        print(f"Model {model.uuid} has no validation predictions!")
    else:
        pprint(val_predictions.head())


def test_confusion_matrix():
    print("\n\n*** Confusion Matrix ***")
    pprint(model.confusion_matrix())


def test_metrics_with_capture_uuid():
    """Test the Performance Metrics using a Capture UUID"""
    metrics = model.get_inference_metrics("auto_inference")
    print("\n\n*** Performance Metrics with Capture UUID ***")
    pprint(metrics)


if __name__ == "__main__":

    # Set Pandas display options
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    # Run the tests
    test_list_inference_runs()
    test_performance_metrics()
    test_retrieval_with_capture_uuid()
    test_validation_predictions()
    test_confusion_matrix()
    test_metrics_with_capture_uuid()
