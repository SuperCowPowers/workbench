"""Tests for the Endpoint functionality (regression + classification)"""

import pandas as pd
import pytest

from workbench.api import Endpoint, FeatureSet
from workbench.utils.endpoint_utils import get_evaluation_data


@pytest.fixture(scope="module")
def reg_endpoint():
    return Endpoint("abalone-regression")


@pytest.fixture(scope="module")
def class_endpoint():
    return Endpoint("wine-classification")


def test_endpoint_exists(reg_endpoint, class_endpoint):
    """Verify the endpoints are deployed and healthy"""
    assert reg_endpoint.exists()
    assert class_endpoint.exists()
    print(f"Endpoint: {reg_endpoint.name}")
    print(f"Created: {reg_endpoint.created()}")
    print(f"Modified: {reg_endpoint.modified()}")
    print(f"Tags: {reg_endpoint.get_tags()}")


def test_regression_test_inference(reg_endpoint):
    pred_df = reg_endpoint.test_inference()
    assert not pred_df.empty
    assert "prediction" in pred_df.columns
    print(f"Test inference shape: {pred_df.shape}")
    print(pred_df.head())


def test_classification_test_inference(class_endpoint):
    pred_df = class_endpoint.test_inference()
    assert not pred_df.empty
    assert "prediction" in pred_df.columns
    print(f"Test inference shape: {pred_df.shape}")
    print(pred_df.head())


def test_manual_inference(reg_endpoint):
    """Run inference on a subset and verify the output contract:
    input columns in --> input columns + prediction columns out"""
    eval_df = get_evaluation_data(reg_endpoint)[:10]
    input_cols = set(eval_df.columns)
    print(f"Input columns: {sorted(input_cols)}")

    pred_df = reg_endpoint.inference(eval_df)
    output_cols = set(pred_df.columns)
    print(f"Output columns: {sorted(output_cols)}")

    assert input_cols.issubset(output_cols), f"Missing input columns: {input_cols - output_cols}"
    assert "prediction" in output_cols
    assert len(pred_df) == len(eval_df)
    print(f"Inference shape: {pred_df.shape}")
    print(pred_df.head())


def test_classification_inference_with_subset_of_labels(class_endpoint):
    eval_data_df = get_evaluation_data(class_endpoint)[:50]

    # Subset the rows to only include the first 2 classes
    print("Using only TypeA and TypeB classes")
    eval_data_df = eval_data_df[eval_data_df["wine_class"].isin(["TypeA", "TypeB"])]
    class_endpoint.inference(eval_data_df)

    # Subset the rows to only include one label
    print("Using only TypeA class")
    eval_data_df = eval_data_df[eval_data_df["wine_class"].isin(["TypeA"])]
    class_endpoint.inference(eval_data_df)

    # Try only one row
    print("Using only one row")
    eval_data_df = eval_data_df.iloc[:1]
    class_endpoint.inference(eval_data_df)

    # Try ZERO rows
    print("Using zero rows")
    eval_data_df = eval_data_df.iloc[:0]
    pred_df = class_endpoint.inference(eval_data_df)
    print(pred_df)


def test_classification_roc_auc(class_endpoint):
    # Compute performance metrics for our test predictions
    eval_data_df = get_evaluation_data(class_endpoint)[:50]
    pred_df = class_endpoint.inference(eval_data_df)

    # Normal test ROCAUC should 1 (or close to 1)
    target_column = "wine_class"
    metrics = class_endpoint.classification_metrics(target_column, pred_df)
    print(metrics)

    # Now switch the prediction probability columns and check the ROCAUC
    temp = pred_df["TypeA_proba"]
    pred_df["TypeA_proba"] = pred_df["TypeB_proba"]
    pred_df["TypeB_proba"] = temp
    metrics = class_endpoint.classification_metrics(target_column, pred_df)
    print(metrics)

    # Okay, now we're going to generate a fake prediction dataframe
    data = {
        "id": [1, 2, 3, 4, 5],
        "target": ["TypeB", "TypeC", "TypeB", "TypeB", "TypeA"],  # True classes
        "prediction": ["TypeA", "TypeA", "TypeB", "TypeC", "TypeC"],  # Mostly wrong predictions
        "TypeB_proba": [0.3, 0.2, 0.4, 0.3, 0.2],
        "TypeC_proba": [0.3, 0.4, 0.3, 0.4, 0.4],
        "TypeA_proba": [0.4, 0.4, 0.3, 0.3, 0.4],
    }
    pred_df = pd.DataFrame(data)
    metrics = class_endpoint.classification_metrics("target", pred_df)
    print(metrics)


@pytest.mark.medium  # ~100s: endpoint inference on 50 rows
def test_regression_metrics(reg_endpoint):
    # Compute performance metrics for our test predictions
    target_column = "class_number_of_rings"
    eval_data_df = get_evaluation_data(reg_endpoint)[:50]
    pred_df = reg_endpoint.inference(eval_data_df)
    metrics = reg_endpoint.regression_metrics(target_column, pred_df)
    print(metrics)

    # Compute residuals for our test predictions
    residuals = reg_endpoint.residuals(target_column, pred_df)
    print(residuals)


def test_classification_metrics(class_endpoint):
    eval_data_df = get_evaluation_data(class_endpoint)[:50]
    pred_df = class_endpoint.inference(eval_data_df)
    print(pred_df)

    # Classification Metrics
    target_column = "wine_class"
    metrics = class_endpoint.classification_metrics(target_column, pred_df)
    print(metrics)

    # Classification Confusion Matrix
    confusion_matrix = class_endpoint.generate_confusion_matrix(target_column, pred_df)
    print(confusion_matrix)

    # What happens if we ask for residuals on a classification endpoint?
    residuals = class_endpoint.residuals(target_column, pred_df)
    print(residuals)


def test_fast_inference(class_endpoint):
    eval_data_df = get_evaluation_data(class_endpoint)[:50]
    pred_df = class_endpoint.fast_inference(eval_data_df)
    print(pred_df)


def test_categorical_features():
    """Test the Categorical Features"""

    # Grab the first row of our test_features
    test_features = FeatureSet("test_features")
    df = test_features.pull_dataframe()[:1]
    end = Endpoint("test-regression")

    # Food is roughly correlated to salary
    for food in ["pizza", "tacos", "steak", "sushi"]:
        df.at[0, "food"] = food
        pred_df = end.inference(df)
        print(f"{pred_df['food']} -> {pred_df['prediction']}")


if __name__ == "__main__":
    reg_end = Endpoint("abalone-regression")
    class_end = Endpoint("wine-classification")

    test_endpoint_exists(reg_end, class_end)
    test_regression_test_inference(reg_end)
    test_classification_test_inference(class_end)
    test_manual_inference(reg_end)
    test_classification_inference_with_subset_of_labels(class_end)
    test_regression_metrics(reg_end)
    test_classification_metrics(class_end)
    test_classification_roc_auc(class_end)
    test_fast_inference(class_end)
    test_categorical_features()

    print("\nAll endpoint tests passed!")
