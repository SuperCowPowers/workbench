"""Tests for the bootstrap helpers in metrics_utils."""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.metrics_utils import bootstrap_compare, bootstrap_metric

MAE = lambda d: float(np.abs(d["y"] - d["pred"]).mean())  # noqa: E731


@pytest.fixture
def frame():
    rng = np.random.default_rng(42)
    y = rng.normal(5.0, 1.0, 300)
    return pd.DataFrame({"y": y, "pred": y + rng.normal(0, 0.5, 300)})


def test_point_estimate_is_the_unresampled_metric(frame):
    assert bootstrap_metric(frame, MAE, n_resamples=100)["value"] == pytest.approx(MAE(frame))


def test_interval_brackets_the_estimate(frame):
    r = bootstrap_metric(frame, MAE, n_resamples=500)
    assert r["ci_lower"] < r["value"] < r["ci_upper"]
    assert r["std"] > 0
    assert r["n"] == len(frame)


def test_seed_makes_it_reproducible(frame):
    a = bootstrap_metric(frame, MAE, n_resamples=200)
    b = bootstrap_metric(frame, MAE, n_resamples=200)
    assert a == b


def test_more_data_narrows_the_interval():
    rng = np.random.default_rng(0)

    def make(n):
        y = rng.normal(5.0, 1.0, n)
        return pd.DataFrame({"y": y, "pred": y + rng.normal(0, 0.5, n)})

    small = bootstrap_metric(make(80), MAE, n_resamples=400)
    large = bootstrap_metric(make(2000), MAE, n_resamples=400)
    assert large["std"] < small["std"]


def test_compare_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(5.0, 1.0, 400)
    idx = [f"m{i}" for i in range(400)]
    good = pd.DataFrame({"y": y, "pred": y + rng.normal(0, 0.2, 400)}, index=idx)
    poor = pd.DataFrame({"y": y, "pred": y + rng.normal(0, 1.2, 400)}, index=idx)

    r = bootstrap_compare(good, poor, MAE, n_resamples=300)
    assert r["delta"] < 0  # lower MAE is better
    assert r["ci_upper"] < 0  # interval excludes zero
    assert r["p_a_better"] > 0.99


def test_compare_reports_a_tie_as_a_tie():
    rng = np.random.default_rng(2)
    y = rng.normal(5.0, 1.0, 400)
    idx = [f"m{i}" for i in range(400)]
    a = pd.DataFrame({"y": y, "pred": y + rng.normal(0, 0.6, 400)}, index=idx)
    b = pd.DataFrame({"y": y, "pred": y + rng.normal(0, 0.6, 400)}, index=idx)

    r = bootstrap_compare(a, b, MAE, n_resamples=300)
    assert r["ci_lower"] < 0 < r["ci_upper"]  # spans zero


def test_lower_is_better_flips_the_win_direction():
    rng = np.random.default_rng(3)
    y = rng.normal(5.0, 1.0, 300)
    idx = [f"m{i}" for i in range(300)]
    good = pd.DataFrame({"y": y, "pred": y + rng.normal(0, 0.2, 300)}, index=idx)
    poor = pd.DataFrame({"y": y, "pred": y + rng.normal(0, 1.2, 300)}, index=idx)

    as_error = bootstrap_compare(good, poor, MAE, n_resamples=200, lower_is_better=True)
    as_score = bootstrap_compare(good, poor, MAE, n_resamples=200, lower_is_better=False)
    assert as_error["p_a_better"] == pytest.approx(1.0 - as_score["p_a_better"])


def test_compare_pairs_on_the_index_and_drops_unpaired():
    rng = np.random.default_rng(4)
    y = rng.normal(5.0, 1.0, 100)
    a = pd.DataFrame({"y": y, "pred": y + 0.1}, index=[f"m{i}" for i in range(100)])
    b = pd.DataFrame({"y": y[:80], "pred": y[:80] + 0.3}, index=[f"m{i}" for i in range(80)])

    r = bootstrap_compare(a, b, MAE, n_resamples=100)
    assert r["n"] == 80


def test_paired_comparison_beats_marginal_intervals():
    """Two models whose own intervals overlap can still be cleanly separated."""
    rng = np.random.default_rng(5)
    y = rng.normal(5.0, 2.0, 400)  # wide spread dominates each model's own interval
    idx = [f"m{i}" for i in range(400)]
    shared = rng.normal(0, 0.9, 400)  # per-row difficulty both models share
    a = pd.DataFrame({"y": y, "pred": y + shared}, index=idx)
    b = pd.DataFrame({"y": y, "pred": y + shared + 0.12}, index=idx)

    ma, mb = bootstrap_metric(a, MAE, n_resamples=300), bootstrap_metric(b, MAE, n_resamples=300)
    assert ma["ci_upper"] > mb["ci_lower"]  # marginal intervals overlap

    r = bootstrap_compare(a, b, MAE, n_resamples=300)
    assert r["ci_upper"] < 0  # paired test still separates them
