"""Tests for multi-target prediction columns."""

import numpy as np
import pandas as pd
import pytest

from workbench.endpoints.multi_target import (
    alias_primary_target,
    apply_per_target_uq,
    multi_target_output_columns,
    per_target_predictions,
)
from workbench.utils.target_transform import Log10

TARGETS = ["ksol", "logd"]


def test_per_target_predictions_columns_and_order():
    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = per_target_predictions(TARGETS, preds)
    assert list(frame.columns) == ["ksol_pred", "ksol_pred_std", "logd_pred", "logd_pred_std"]
    np.testing.assert_allclose(frame["ksol_pred"], [1.0, 3.0])
    np.testing.assert_allclose(frame["logd_pred"], [2.0, 4.0])


def test_per_target_predictions_zero_spread_without_an_ensemble():
    frame = per_target_predictions(TARGETS, np.array([[1.0, 2.0]]))
    assert frame["ksol_pred_std"].tolist() == [0.0]
    assert frame["logd_pred_std"].tolist() == [0.0]


def test_per_target_predictions_spread_from_the_ensemble():
    stack = np.array([[[1.0, 10.0]], [[3.0, 10.0]]])
    frame = per_target_predictions(TARGETS, stack.mean(axis=0), stack)
    assert frame["ksol_pred_std"][0] == pytest.approx(1.0)
    assert frame["logd_pred_std"][0] == pytest.approx(0.0)


def test_per_target_predictions_inverts_only_the_transformed_target():
    """Each target carries its own transform; an omitted target stays untouched."""
    preds = np.array([[2.0, 2.0]])
    frame = per_target_predictions(TARGETS, preds, transforms={"ksol": Log10()})
    assert frame["ksol_pred"][0] == pytest.approx(100.0)
    assert frame["logd_pred"][0] == pytest.approx(2.0)


def test_per_target_predictions_inverts_the_mean_not_the_members():
    """The point prediction is the inverted mean (the median), not the mean of inverses."""
    stack = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    frame = per_target_predictions(TARGETS, stack.mean(axis=0), stack, transforms={"ksol": Log10()})
    assert frame["ksol_pred"][0] == pytest.approx(100.0), "10**mean([1, 3]) == 100"
    mean_of_inverses = (10.0 + 1000.0) / 2
    assert frame["ksol_pred"][0] != pytest.approx(mean_of_inverses)


def test_per_target_predictions_spread_is_in_original_units():
    stack = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    frame = per_target_predictions(TARGETS, stack.mean(axis=0), stack, transforms={"ksol": Log10()})
    assert frame["ksol_pred_std"][0] == pytest.approx(np.std([10.0, 1000.0]))


def test_per_target_predictions_clips_bounds():
    preds = np.array([[-5.0, 200.0]])
    frame = per_target_predictions(TARGETS, preds, bounds={"ksol": (0.0, 100.0), "logd": (0.0, 100.0)})
    assert frame["ksol_pred"][0] == 0.0
    assert frame["logd_pred"][0] == 100.0


def test_alias_primary_target():
    df = pd.DataFrame({"ksol_pred": [1.0], "ksol_pred_std": [0.5], "ksol_confidence": [0.9]})
    alias_primary_target(df, "ksol", ["confidence", "q_05"])
    assert df["prediction"][0] == 1.0
    assert df["prediction_std"][0] == 0.5
    assert df["confidence"][0] == 0.9
    assert "q_05" not in df.columns, "a UQ name the frame lacks is skipped"


def test_apply_per_target_uq_whole_columns():
    df = pd.DataFrame({"id": [1, 2]})
    uq_outs = {
        "ksol": pd.DataFrame({"confidence": [0.1, 0.2]}),
        "logd": pd.DataFrame({"confidence": [0.3, 0.4]}),
    }
    apply_per_target_uq(df, uq_outs, primary="ksol")
    assert df["ksol_confidence"].tolist() == [0.1, 0.2]
    assert df["logd_confidence"].tolist() == [0.3, 0.4]
    assert df["confidence"].tolist() == [0.1, 0.2], "primary is aliased unprefixed"


def test_apply_per_target_uq_with_a_shared_mask():
    df = pd.DataFrame({"id": [1, 2, 3]})
    mask = np.array([True, False, True])
    apply_per_target_uq(df, {"ksol": pd.DataFrame({"confidence": [0.1, 0.9]})}, primary="ksol", masks=mask)
    assert df["ksol_confidence"].tolist()[0] == 0.1
    assert np.isnan(df["ksol_confidence"][1]), "unmasked rows stay NaN"
    assert df["ksol_confidence"].tolist()[2] == 0.9


def test_apply_per_target_uq_with_per_target_masks():
    """Each target calibrates on its own labeled rows, so masks differ per target."""
    df = pd.DataFrame({"id": [1, 2]})
    masks = {"ksol": np.array([True, False]), "logd": np.array([False, True])}
    uq_outs = {"ksol": pd.DataFrame({"confidence": [0.1]}), "logd": pd.DataFrame({"confidence": [0.4]})}
    apply_per_target_uq(df, uq_outs, primary="ksol", masks=masks)
    assert df["ksol_confidence"][0] == 0.1 and np.isnan(df["ksol_confidence"][1])
    assert np.isnan(df["logd_confidence"][0]) and df["logd_confidence"][1] == 0.4


def test_multi_target_output_columns_order():
    df = pd.DataFrame(
        {
            "id": [1],
            "ksol": [1.0],
            "logd": [2.0],
            "ksol_pred": [1.0],
            "logd_pred": [2.0],
            "ksol_pred_std": [0.1],
            "logd_pred_std": [0.2],
            "prediction": [1.0],
            "prediction_std": [0.1],
            "confidence": [0.9],
            "ksol_confidence": [0.9],
            "logd_confidence": [0.8],
            "unrelated": [0],
        }
    )
    columns = multi_target_output_columns(df, "id", TARGETS)
    assert columns == [
        "id",
        "ksol",
        "logd",
        "ksol_pred",
        "logd_pred",
        "ksol_pred_std",
        "logd_pred_std",
        "prediction",
        "prediction_std",
        "confidence",
        "ksol_confidence",
        "logd_confidence",
    ]
    assert "unrelated" not in columns


def test_multi_target_output_columns_drops_absent_columns():
    """One order serves both the out-of-fold and held-out files."""
    df = pd.DataFrame({"id": [1], "ksol": [1.0], "ksol_pred": [1.0], "ksol_pred_std": [0.1]})
    assert multi_target_output_columns(df, "id", ["ksol"]) == ["id", "ksol", "ksol_pred", "ksol_pred_std"]
