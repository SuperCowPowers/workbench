"""Tests for the UQ metrics, which score the error correlations in the model's space."""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.model_utils import target_uq_frame, uq_metrics
from workbench.utils.target_transform import Log10

QUANTILES = {"q_025": -1.96, "q_05": -1.645, "q_10": -1.282, "q_16": -1.0, "q_25": -0.674}
QUANTILES.update({"q_75": 0.674, "q_84": 1.0, "q_90": 1.282, "q_95": 1.645, "q_975": 1.96})


@pytest.fixture
def log_model_frame():
    """A log10 model's output frame: calibrated in log units, reported in original units."""
    rng = np.random.default_rng(0)
    n = 400
    log_pred = rng.uniform(0, 5, n)
    sigma = rng.uniform(0.05, 0.6, n)
    log_truth = log_pred + rng.normal(scale=sigma)
    df = pd.DataFrame({"target": 10**log_truth, "prediction": 10**log_pred, "prediction_std": sigma})
    df["confidence"] = 1.0 - pd.Series(sigma).rank(pct=True)
    for name, z in QUANTILES.items():
        df[name] = 10 ** (log_pred + z * sigma)
    return df


def test_correlations_mirror_each_other_in_the_model_space(log_model_frame):
    metrics = uq_metrics(log_model_frame, "target", Log10())
    assert metrics["interval_to_error_corr"] == pytest.approx(-metrics["confidence_to_error_corr"], abs=1e-9)
    assert metrics["interval_to_error_corr"] > 0.3


def test_transform_leaves_coverage_and_widths_on_the_reported_scale(log_model_frame):
    plain = uq_metrics(log_model_frame, "target")
    scored = uq_metrics(log_model_frame, "target", Log10())
    for key in ("coverage_68", "coverage_95", "median_width_68", "median_width_95", "n_samples"):
        assert scored[key] == plain[key]
    assert plain["interval_to_error_corr"] != pytest.approx(scored["interval_to_error_corr"], abs=0.01)


def test_values_outside_the_transform_domain_are_left_out(log_model_frame):
    log_model_frame.loc[0, "q_16"] = 0.0
    metrics = uq_metrics(log_model_frame, "target", Log10())
    assert np.isfinite(metrics["interval_to_error_corr"]) and metrics["n_samples"] == len(log_model_frame)


def test_target_uq_frame_carries_the_targets_own_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, np.nan], "a_pred": [1.1, 2.1], "b_pred": [5.5, 6.5]})
    df = df.assign(a_pred_std=0.1, b_pred_std=0.5, prediction=df["a_pred"], prediction_std=0.1)
    df = df.assign(confidence=0.9, a_confidence=0.9, b_confidence=0.2)
    df = df.assign(q_16=1.0, a_q_16=1.0, b_q_16=5.0, q_84=2.2)

    view = target_uq_frame(df, "b")
    assert len(view) == 1, "only the target's labeled rows"
    assert view["prediction"].iloc[0] == 5.5 and view["prediction_std"].iloc[0] == 0.5
    assert view["confidence"].iloc[0] == 0.2 and view["q_16"].iloc[0] == 5.0
    assert "q_84" not in view.columns, "another target's quantile is never left behind"
