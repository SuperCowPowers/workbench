"""Tests for the per-model training-view row roles: ``exclude_ids`` and ``validation_ids``.

Row role is orthogonal to ``sample_weight`` (a pure framework weight): weight 0 no
longer drops a row. ``exclude_ids`` drops rows from the view entirely;
``validation_ids`` keeps rows but marks them held-out (scored, never trained).
"""

import pytest

# Workbench Imports
from workbench.api import FeatureSet, ModelFramework
from workbench.core.artifacts.model_core import ModelType


@pytest.mark.long
def test_exclude_ids_exclusion():
    """``exclude_ids`` drops rows from the model's training view entirely.

    Marks 50 rows as excluded, verifies those rows are absent from the created
    model's training view (while the rest are retained), then runs an inference
    pass on the held-out 50 to confirm they're still scorable.

    Note: exclusion is now its own role. ``sample_weight`` is a pure framework
    weight (weight 0 no longer drops a row); use ``exclude_ids`` to drop rows and
    ``validation_ids`` to hold rows out as a scored validation set.
    """
    fs = FeatureSet("abalone_features")
    id_column = fs.id_column

    # Pick 50 ids to exclude from training
    full_df = fs.pull_dataframe()
    all_ids = set(full_df[id_column].tolist())
    excluded_ids = list(all_ids)[:50]

    # Create the model with those rows excluded from training
    name = "abalone-regression-weights-test"
    model = fs.to_model(
        name=name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="class_number_of_rings",
        tags=["test", "abalone", "weights-test"],
        description="Abalone Regression (exclude_ids test)",
        exclude_ids=excluded_ids,
    )

    try:
        # The 50 excluded rows must NOT appear in the model's training view
        training_ids = set(model.training_view().pull_dataframe()[id_column].tolist())
        assert training_ids.isdisjoint(excluded_ids), "Excluded rows leaked into the training view!"

        # ...and the non-excluded rows must still be present
        retained_sample = [row_id for row_id in all_ids if row_id not in set(excluded_ids)][:10]
        assert all(row_id in training_ids for row_id in retained_sample)

        # The held-out rows should still be scorable via an inference run
        endpoint = model.to_endpoint(name=name, tags=["test", "abalone", "weights-test"])
        try:
            held_out_df = full_df[full_df[id_column].isin(excluded_ids)]
            pred_df = endpoint.inference(held_out_df)
            assert len(pred_df) == 50
            assert "prediction" in pred_df.columns
        finally:
            endpoint.delete()
    finally:
        model.delete()


@pytest.mark.long
def test_validation_ids_holdout():
    """``validation_ids`` keeps rows in the view but marks them held-out.

    Marks 50 rows as validation, verifies those rows ARE retained in the model's
    training view (unlike exclude) and carry ``validation = True``, while the rest
    carry ``validation = False``. The held-out rows are scored (not trained) by the
    model script — surfaced as ``validation``-marked rows in validation_predictions.
    """
    fs = FeatureSet("abalone_features")
    id_column = fs.id_column

    # Pick 50 ids to hold out as a validation set
    full_df = fs.pull_dataframe()
    all_ids = full_df[id_column].tolist()
    validation_ids = all_ids[:50]

    name = "abalone-regression-validation-test"
    model = fs.to_model(
        name=name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="class_number_of_rings",
        tags=["test", "abalone", "validation-test"],
        description="Abalone Regression (validation_ids test)",
        validation_ids=validation_ids,
    )

    try:
        view_df = model.training_view().pull_dataframe()
        view_ids = set(view_df[id_column].tolist())

        # Validation rows are RETAINED in the view (kept, not dropped)
        assert set(validation_ids).issubset(view_ids), "Validation rows were dropped from the training view!"

        # The `validation` marker column separates held-out rows from the rest
        assert "validation" in view_df.columns, "Training view is missing the `validation` column"
        val_flag = view_df.set_index(id_column)["validation"].astype(bool)
        assert val_flag.loc[validation_ids].all(), "Validation rows are not marked `validation = True`"
        non_val = [row_id for row_id in all_ids if row_id not in set(validation_ids)]
        assert not val_flag.loc[non_val].any(), "Non-validation rows are incorrectly marked `validation = True`"
    finally:
        model.delete()


if __name__ == "__main__":
    test_exclude_ids_exclusion()
    test_validation_ids_holdout()
