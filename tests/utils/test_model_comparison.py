"""Tests for the model_comparison utility (champion vs challenger metrics)"""

import pytest

# Workbench Imports
from workbench.cached.cached_model import CachedModel
import pandas as pd

# Workbench Imports
from workbench.api import Reports
from workbench.utils.model_comparison import (
    _contested,
    contest_ranking,
    contest_report,
    contest_summaries,
    model_comparison,
    prediction_comparison,
    rank_models,
)


def _contest(champ_value, challengers, metric="rmse"):
    """(champ_row, chall_rows) shaped like rank_models()/contest_ranking() output.

    challengers: [(name, value)] already ranked best-first, as contest_ranking() returns them.
    Δ is metrics-aware and absolute: champion - challenger for rmse (lower is better),
    challenger - champion for f1. Positive Δ always means the challenger is better.
    """
    champ_row = pd.DataFrame([{metric: champ_value}], index=["champ"])
    deltas = [(champ_value - v) if metric in ("rmse", "mae", "medae") else (v - champ_value) for _, v in challengers]
    chall_rows = pd.DataFrame(
        [{metric: v, f"Δ{metric}": d} for (_, v), d in zip(challengers, deltas)],
        index=[n for n, _ in challengers],
    )
    return champ_row, chall_rows


def test_regression_comparison():
    """Regressor comparison: [a, b, delta] rows with metrics-aware delta signs"""
    comp = model_comparison(CachedModel("aqsol-regression"), CachedModel("aqsol-regression-1"), "full_cross_fold")
    assert list(comp.index) == ["aqsol-regression", "aqsol-regression-1", "delta"]
    assert {"rmse", "mae", "r2", "spearmanr", "support"} <= set(comp.columns)

    # Metrics-aware: positive delta always means model_b is better
    row_a, row_b, delta = comp.iloc[0], comp.iloc[1], comp.loc["delta"]
    assert delta["rmse"] == pytest.approx(row_a["rmse"] - row_b["rmse"])  # lower is better
    assert delta["r2"] == pytest.approx(row_b["r2"] - row_a["r2"])  # higher is better


def test_classification_comparison():
    """Classifier comparison uses the 'all' summary row"""
    comp = model_comparison(CachedModel("aqsol-class"), CachedModel("aqsol-class-1"), "full_cross_fold")
    assert list(comp.index) == ["aqsol-class", "aqsol-class-1", "delta"]
    assert {"precision", "recall", "f1", "roc_auc"} <= set(comp.columns)


def test_missing_run_returns_none():
    """A missing inference run on either model returns None"""
    assert model_comparison(CachedModel("aqsol-regression"), CachedModel("aqsol-regression-1"), "no-such-run") is None


def test_rank_models():
    """rank_models() sorts regressors by rmse (low to high)"""
    models = [CachedModel("aqsol-regression-1"), CachedModel("aqsol-regression-2")]
    ranked = rank_models(models, "full_cross_fold")
    assert list(ranked["rmse"]) == sorted(ranked["rmse"])


def test_contest_ranking():
    """contest_ranking() ranks challengers with metrics-aware Δ columns vs the champion"""
    champion = CachedModel("aqsol-regression")
    challengers = [CachedModel("aqsol-regression-1"), CachedModel("aqsol-regression-2")]
    ranked = contest_ranking(champion, challengers, "full_cross_fold")
    assert list(ranked.columns[:2]) == ["rmse", "Δrmse"]  # Δ interleaved after each metric
    assert "Δsupport" not in ranked.columns

    # Δrmse is metrics-aware: champion rmse minus challenger rmse (positive = challenger better)
    champ_rmse = champion.get_inference_metrics("full_cross_fold").iloc[0]["rmse"]
    for name, row in ranked.iterrows():
        assert row["Δrmse"] == pytest.approx(champ_rmse - row["rmse"])


def test_contest_report():
    """contest_report() has champion first (Δ=0), ranked challengers, and contest metadata columns"""
    champion = CachedModel("aqsol-regression")
    challengers = [CachedModel("aqsol-regression-1"), CachedModel("aqsol-regression-2")]
    report = contest_report(champion, challengers, "aqsol-regression", "full_cross_fold")

    assert list(report["role"]) == ["champion", "challenger", "challenger"]
    assert report["model"].iloc[0] == "aqsol-regression"
    assert report["endpoint"].eq("aqsol-regression").all()
    assert report["framework"].isin(["xgboost", "pytorch", "chemprop", "hybrid", "multi-task", "sklearn"]).all()
    assert report["inference_run"].eq("full_cross_fold").all()
    assert report.loc[0, "Δrmse"] == 0.0  # champion delta vs itself
    assert report["created"].notna().all()
    assert report["contested"].nunique() == 1  # contest-level flag, repeated on every row

    # Challengers ranked best-first (regressor: rmse low to high)
    chall_rmse = list(report.loc[report["role"] == "challenger", "rmse"])
    assert chall_rmse == sorted(chall_rmse)


def test_contested_skips_the_champions_twin():
    """The champion is a frozen copy of a challenger, so its twin sits at Δ=0 and must not
    make the contest contested by itself (otherwise every promoted contest is contested)"""
    # Twin at Δ=0, next real challenger clearly worse (-8%)
    champ, chall = _contest(0.50, [("twin", 0.50), ("worse", 0.54)])
    assert _contested(champ, chall) is False

    # Twin at Δ=0, next real challenger close (-0.5%) -> the twin is skipped, the real one counts
    champ, chall = _contest(0.50, [("twin", 0.50), ("close", 0.5025)])
    assert _contested(champ, chall) is True

    # Every challenger is a twin -> nothing real to contest against
    champ, chall = _contest(0.50, [("twin-a", 0.50), ("twin-b", 0.50)])
    assert _contested(champ, chall) is False


def test_contested_percent_threshold():
    """CONTESTED_PCT is a percent of the champion's value (Δ is absolute) and the rule is
    'better, or at most 1% worse'"""
    # Just inside the 1% band (-0.9%) vs just outside (-1.1%)
    champ, chall = _contest(0.50, [("inside", 0.5045)])
    assert _contested(champ, chall) is True
    champ, chall = _contest(0.50, [("outside", 0.5055)])
    assert _contested(champ, chall) is False

    # A challenger that beats the champion but wasn't promoted: blocked/broken pipeline
    champ, chall = _contest(0.50, [("better", 0.475)])
    assert _contested(champ, chall) is True

    # Percent is relative, so the same absolute Δ flips with the champion's scale
    champ, chall = _contest(100.0, [("tiny-abs-delta", 100.4)])  # -0.4%
    assert _contested(champ, chall) is True


def test_contested_classifier_and_edges():
    """Classifiers rank on f1 (higher is better); degenerate inputs are not contested"""
    # f1 challenger 0.25% worse -> inside the band
    champ, chall = _contest(0.80, [("twin", 0.80), ("close", 0.798)], metric="f1")
    assert _contested(champ, chall) is True
    # f1 challenger 5% worse -> outside
    champ, chall = _contest(0.80, [("worse", 0.76)], metric="f1")
    assert _contested(champ, chall) is False

    # No challengers, and a zero-valued champion (no meaningful percent)
    champ, _ = _contest(0.50, [("x", 0.49)])
    assert _contested(champ, pd.DataFrame()) is False
    champ, chall = _contest(0.0, [("x", 0.01)])
    assert _contested(champ, chall) is False

    # Champion metrics missing entirely
    assert _contested(pd.DataFrame(), chall) is False


def test_contest_summaries():
    """contest_summaries() rolls each published /contests/* report into one card row"""
    report = pd.DataFrame(
        {
            "model": ["champ", "chall-a", "chall-b"],
            "role": ["champion", "challenger", "challenger"],
            "endpoint": "zzz-test-contest",
            "rmse": [0.50, 0.45, 0.60],
            "Δrmse": [0.0, 0.05, -0.10],
            "inference_run": "full_cross_fold",
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "contested": True,
        }
    )
    reports = Reports()
    reports.upsert("/contests/zzz-test-contest", report)
    try:
        summaries = contest_summaries()
        row = summaries[summaries["endpoint"] == "zzz-test-contest"].iloc[0]
        assert row["champion"] == "champ"
        assert row["challengers"] == 2
        assert row["top_challenger"] == "chall-a"
        assert row["top_delta"] == pytest.approx(0.05)
        assert bool(row["contested"]) is True
    finally:
        reports.delete("/contests/zzz-test-contest")


def test_prediction_comparison():
    """prediction_comparison() stacks both models' predictions with a 'model' column"""
    preds = prediction_comparison(CachedModel("aqsol-regression"), CachedModel("aqsol-regression-2"), "full_cross_fold")
    assert list(preds["model"].unique()) == ["aqsol-regression", "aqsol-regression-2"]
    assert {"prediction", "solubility"} <= set(preds.columns)


if __name__ == "__main__":
    test_regression_comparison()
    test_classification_comparison()
    test_missing_run_returns_none()
    test_rank_models()
    test_contest_ranking()
    test_contest_report()
    test_contested_skips_the_champions_twin()
    test_contested_percent_threshold()
    test_contested_classifier_and_edges()
    test_contest_summaries()
    test_prediction_comparison()
    print("All model_comparison tests passed!")
