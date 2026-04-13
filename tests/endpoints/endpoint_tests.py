"""Tests for normal Endpoint inference (abalone-regression)"""

from workbench.api import Endpoint
from workbench.utils.endpoint_utils import get_evaluation_data

endpoint = Endpoint("abalone-regression")


def test_endpoint_exists():
    """Verify the endpoint is deployed and healthy"""
    assert endpoint.exists()
    print(f"Endpoint: {endpoint.name}")
    print(f"Created: {endpoint.created()}")
    print(f"Modified: {endpoint.modified()}")


def test_auto_inference():
    """Run auto_inference and verify we get predictions back"""
    pred_df = endpoint.auto_inference()
    assert not pred_df.empty
    assert "prediction" in pred_df.columns
    print(f"Auto inference shape: {pred_df.shape}")
    print(pred_df.head())


def test_manual_inference():
    """Run inference on a subset and verify the output contract:
    input columns in --> input columns + prediction columns out"""
    eval_df = get_evaluation_data(endpoint)[:10]
    input_cols = set(eval_df.columns)
    print(f"Input columns: {sorted(input_cols)}")

    pred_df = endpoint.inference(eval_df)
    output_cols = set(pred_df.columns)
    print(f"Output columns: {sorted(output_cols)}")

    # All input columns should still be present
    assert input_cols.issubset(output_cols), f"Missing input columns: {input_cols - output_cols}"

    # Prediction column should be added
    assert "prediction" in output_cols

    # Row count should be preserved
    assert len(pred_df) == len(eval_df)
    print(f"Inference shape: {pred_df.shape}")
    print(pred_df.head())


if __name__ == "__main__":
    test_endpoint_exists()
    test_auto_inference()
    test_manual_inference()
    print("\nAll endpoint tests passed!")
