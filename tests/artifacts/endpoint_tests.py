"""Tests for the Endpoint functionality"""

# SageWorks Imports
from sageworks.core.artifacts.endpoint_core import EndpointCore
from sageworks.utils.endpoint_utils import fs_evaluation_data

reg_endpoint = EndpointCore("abalone-regression-end")
class_endpoint = EndpointCore("wine-classification-end")


def test_general_info():
    """Simple test of the Endpoint functionality"""

    # Call the various methods

    # Let's do a check/validation of the Endpoint
    assert reg_endpoint.exists()

    # Creation/Modification Times
    print(reg_endpoint.created())
    print(reg_endpoint.modified())

    # Get the tags associated with this Endpoint
    print(f"Tags: {reg_endpoint.get_tags()}")


def test_regression_auto_inference():
    pred_df = reg_endpoint.auto_inference()
    print(pred_df)


def test_classification_auto_inference():
    pred_df = class_endpoint.auto_inference()
    print(pred_df)


def test_manual_inference():
    eval_data_df = fs_evaluation_data(reg_endpoint)[:50]
    pred_df = reg_endpoint.inference(eval_data_df)
    print(pred_df)


def test_regression_metrics():
    # Compute performance metrics for our test predictions
    target_column = "class_number_of_rings"
    eval_data_df = fs_evaluation_data(reg_endpoint)[:50]
    pred_df = reg_endpoint.inference(eval_data_df)
    metrics = reg_endpoint.regression_metrics(target_column, pred_df)
    print(metrics)

    # Compute residuals for our test predictions
    residuals = reg_endpoint.residuals(target_column, pred_df)
    print(residuals)


def test_classification_metrics():
    eval_data_df = fs_evaluation_data(class_endpoint)[:50]
    pred_df = class_endpoint.inference(eval_data_df)
    print(pred_df)

    # Classification Metrics
    target_column = "wine_class"
    metrics = class_endpoint.classification_metrics(target_column, pred_df)
    print(metrics)

    # Classification Confusion Matrix
    confusion_matrix = class_endpoint.confusion_matrix(target_column, pred_df)
    print(confusion_matrix)

    # What happens if we ask for residuals on a classification endpoint?
    residuals = class_endpoint.residuals(target_column, pred_df)
    print(residuals)


if __name__ == "__main__":

    # Run the tests
    test_general_info()
    test_regression_auto_inference()
    test_classification_auto_inference()
    test_manual_inference()
    test_regression_metrics()
    test_classification_metrics()

    print("All tests passed!")
