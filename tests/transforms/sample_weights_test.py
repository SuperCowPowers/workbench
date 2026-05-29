"""Test that per-model sample weights exclude rows from the training view"""

import pytest

# Workbench Imports
from workbench.api import FeatureSet, ModelFramework
from workbench.core.artifacts.model_core import ModelType


@pytest.mark.long
def test_sample_weights_exclusion():
    """Sample weights of 0 exclude rows from the model's training view.

    Sets sample_weight=0 on 50 rows, verifies those rows are absent from the
    created model's training view (while the rest are retained), then runs an
    inference pass on the held-out 50 to confirm they're still scorable.
    """
    fs = FeatureSet("abalone_features")
    id_column = fs.id_column

    # Pick 50 ids to exclude from training (weight 0)
    full_df = fs.pull_dataframe()
    all_ids = set(full_df[id_column].tolist())
    excluded_ids = list(all_ids)[:50]
    sample_weights = {row_id: 0.0 for row_id in excluded_ids}

    # Create the model with those rows excluded from training
    name = "abalone-regression-weights-temp"
    model = fs.to_model(
        name=name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="class_number_of_rings",
        tags=["temp", "abalone", "weights-test"],
        description="Abalone Regression (sample-weight exclusion test)",
        sample_weights=sample_weights,
    )

    try:
        # The 50 weight-0 rows must NOT appear in the model's training view
        training_ids = set(model.training_view().pull_dataframe()[id_column].tolist())
        assert training_ids.isdisjoint(excluded_ids), "Weight-0 rows leaked into the training view!"

        # ...and the non-excluded rows must still be present
        retained_sample = [row_id for row_id in all_ids if row_id not in set(excluded_ids)][:10]
        assert all(row_id in training_ids for row_id in retained_sample)

        # The held-out rows should still be scorable via an inference run
        endpoint = model.to_endpoint(name=name, tags=["temp", "abalone", "weights-test"])
        try:
            held_out_df = full_df[full_df[id_column].isin(excluded_ids)]
            pred_df = endpoint.inference(held_out_df)
            assert len(pred_df) == 50
            assert "prediction" in pred_df.columns
        finally:
            endpoint.delete()
    finally:
        model.delete()


if __name__ == "__main__":
    test_sample_weights_exclusion()
