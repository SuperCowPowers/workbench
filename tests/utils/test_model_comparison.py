"""Tests for the model_comparison utility (champion vs challenger metrics)"""

import pytest

# Workbench Imports
from workbench.cached.cached_model import CachedModel
from workbench.utils.model_comparison import contest_ranking, model_comparison, prediction_comparison, rank_models


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
    test_prediction_comparison()
    print("All model_comparison tests passed!")
