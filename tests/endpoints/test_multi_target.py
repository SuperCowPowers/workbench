"""Tests for multi-target prediction columns."""

import json

import numpy as np
import pandas as pd
import pytest

from workbench.endpoints.multi_target import (
    alias_primary_target,
    apply_per_target_uq,
    multi_target_output_columns,
    per_target_predictions,
    report_targets,
)
from workbench.utils.target_transform import Log10, TargetAdapter, load_target_adapter, resolve_target_adapters

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


def test_report_targets_inverts_only_the_transformed_target():
    """Each target carries its own adapter; an untransformed target stays untouched."""
    frame = per_target_predictions(TARGETS, np.array([[2.0, 2.0]]))
    alias_primary_target(frame, "ksol")
    report_targets(frame, {"ksol": TargetAdapter(Log10()), "logd": TargetAdapter()}, primary="ksol")
    assert frame["ksol_pred"][0] == pytest.approx(100.0)
    assert frame["prediction"][0] == pytest.approx(100.0), "the primary alias is reported too"
    assert frame["logd_pred"][0] == pytest.approx(2.0)


def test_report_targets_inverts_the_mean_not_the_members():
    """The point prediction is the inverted mean (the median), not the mean of inverses."""
    stack = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    frame = per_target_predictions(TARGETS, stack.mean(axis=0), stack)
    alias_primary_target(frame, "ksol")
    report_targets(frame, {"ksol": TargetAdapter(Log10()), "logd": TargetAdapter()}, primary="ksol")
    assert frame["ksol_pred"][0] == pytest.approx(100.0), "10**mean([1, 3]) == 100"
    assert frame["ksol_pred"][0] != pytest.approx((10.0 + 1000.0) / 2)


def test_report_targets_leaves_the_spread_in_model_space():
    stack = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    frame = per_target_predictions(TARGETS, stack.mean(axis=0), stack)
    alias_primary_target(frame, "ksol")
    report_targets(frame, {"ksol": TargetAdapter(Log10()), "logd": TargetAdapter()}, primary="ksol")
    assert frame["ksol_pred_std"][0] == pytest.approx(np.std([1.0, 3.0]))
    assert frame["prediction_std"][0] == pytest.approx(np.std([1.0, 3.0]))


def test_report_targets_clips_bounds():
    frame = per_target_predictions(TARGETS, np.array([[-5.0, 200.0]]))
    alias_primary_target(frame, "ksol")
    adapters = {t: TargetAdapter(bounds=(0.0, 100.0)) for t in TARGETS}
    report_targets(frame, adapters, primary="ksol")
    assert frame["ksol_pred"][0] == 0.0
    assert frame["logd_pred"][0] == 100.0


def test_report_targets_inverts_training_truth_only_when_asked():
    frame = pd.DataFrame({"ksol": [2.0], "logd": [1.0], "ksol_pred": [2.0], "logd_pred": [1.0], "prediction": [2.0]})
    adapters = {"ksol": TargetAdapter(Log10()), "logd": TargetAdapter()}
    held_out = frame.copy()
    report_targets(frame, adapters, primary="ksol", truth=True)
    report_targets(held_out, adapters, primary="ksol")
    assert frame["ksol"][0] == pytest.approx(100.0)
    assert held_out["ksol"][0] == 2.0, "held-out rows carry an original-scale target"


def test_report_targets_reports_per_target_and_aliased_quantiles():
    frame = pd.DataFrame({"ksol_pred": [2.0], "logd_pred": [1.0], "prediction": [2.0]})
    uq_outs = {"ksol": pd.DataFrame({"q_95": [3.0]}), "logd": pd.DataFrame({"q_95": [3.0]})}
    apply_per_target_uq(frame, uq_outs, primary="ksol")
    report_targets(frame, {"ksol": TargetAdapter(Log10()), "logd": TargetAdapter()}, primary="ksol")
    assert frame["ksol_q_95"][0] == pytest.approx(1000.0)
    assert frame["q_95"][0] == pytest.approx(1000.0)
    assert frame["logd_q_95"][0] == 3.0


def test_per_target_predictions_on_an_empty_batch():
    """A batch with no parseable molecules still returns the same columns."""
    stack = np.empty((3, 0, 2))
    frame = per_target_predictions(TARGETS, np.mean(stack, axis=0), stack)
    assert list(frame.columns) == ["ksol_pred", "ksol_pred_std", "logd_pred", "logd_pred_std"]
    assert len(frame) == 0


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


def test_training_to_inference_round_trip(tmp_path):
    """The seam the templates sit on: what training writes, inference must read back.

    Training forwards the target and saves the spec; model_fn reads the spec from that file
    and predict_fn reports with it. A perfect model's predictions must therefore come back
    equal to the original-scale targets.
    """
    truth = np.array([[0.001, 2.0], [0.5, 3.0], [250.0, 4.0]])
    hyperparameters = {"target_transform": {"ksol": {"name": "log10", "offset": 0.0001}}, "n_folds": 5}

    # Training: forward the target, then persist the spec the way a template does
    fitted = resolve_target_adapters(hyperparameters, TARGETS)
    trained_on = np.column_stack([fitted[t].forward(truth[:, i]) for i, t in enumerate(TARGETS)])
    (tmp_path / "hyperparameters.json").write_text(json.dumps(hyperparameters))

    # Inference: resolve from the saved file alone, then report a perfect model's output
    adapters = load_target_adapter(str(tmp_path), TARGETS)
    frame = per_target_predictions(TARGETS, trained_on, np.stack([trained_on]))
    alias_primary_target(frame, "ksol")
    report_targets(frame, adapters, primary="ksol")

    np.testing.assert_allclose(frame["ksol_pred"], truth[:, 0], rtol=1e-9)
    np.testing.assert_allclose(frame["logd_pred"], truth[:, 1], rtol=1e-9)


def test_training_to_inference_round_trip_without_a_spec(tmp_path):
    """A model saved with no transform must serve its predictions unchanged."""
    truth = np.array([[1.5, 2.0]])
    (tmp_path / "hyperparameters.json").write_text(json.dumps({"n_folds": 5}))
    adapters = load_target_adapter(str(tmp_path), TARGETS)
    frame = per_target_predictions(TARGETS, truth, None)
    alias_primary_target(frame, "ksol")
    report_targets(frame, adapters, primary="ksol")
    np.testing.assert_allclose(frame["ksol_pred"], truth[:, 0])
    np.testing.assert_allclose(frame["logd_pred"], truth[:, 1])


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


def test_report_targets_clips_quantiles_to_bounds():
    """A target's quantiles are clipped the way its point prediction is."""
    df = pd.DataFrame({"id": [1], "ksol_pred": [50.0], "logd_pred": [50.0], "prediction": [50.0]})
    uq_outs = {
        "ksol": pd.DataFrame({"q_05": [-5.0], "q_95": [150.0], "prediction_std": [60.0]}),
        "logd": pd.DataFrame({"q_05": [-5.0], "q_95": [150.0], "prediction_std": [60.0]}),
    }
    apply_per_target_uq(df, uq_outs, "ksol")
    report_targets(df, {"ksol": TargetAdapter(bounds=(0.0, 100.0)), "logd": TargetAdapter()}, primary="ksol")

    assert df["ksol_q_05"][0] == 0.0 and df["ksol_q_95"][0] == 100.0
    assert df["q_05"][0] == 0.0, "the primary target's aliases carry the clipped values"
    assert df["ksol_prediction_std"][0] == 60.0, "a dispersion estimate is not clipped"
    assert df["logd_q_05"][0] == -5.0, "an unbounded target is untouched"


def test_apply_per_target_uq_without_bounds_leaves_quantiles_alone():
    df = pd.DataFrame({"id": [1]})
    uq_outs = {"ksol": pd.DataFrame({"q_95": [150.0]})}
    apply_per_target_uq(df, uq_outs, "ksol")
    assert df["ksol_q_95"][0] == 150.0
