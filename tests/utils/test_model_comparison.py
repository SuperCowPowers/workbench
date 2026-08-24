"""Tests for the model_comparison utility (champion vs challenger metrics)

Contests evolve: a promotion repoints the champion endpoint at the winning model, and the
promotion node accumulates challengers. So the models under test are resolved by lineage —
the champion is whatever model the endpoint currently serves, and the challengers are
whatever the promotion node currently lists — never by assuming a model name.
"""

import pytest

# Workbench Imports
from workbench.api import Endpoint
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_model import CachedModel
import pandas as pd

# Workbench Imports
from workbench.utils.model_comparison import (
    _contested,
    _binary_acc,
    contest_ranking,
    contest_report,
    model_comparison,
    prediction_comparison,
    rank_models,
)

REGRESSION_ENDPOINT = "aqsol-regression"
CLASSIFICATION_ENDPOINT = "aqsol-class"


def _champion(endpoint_name: str) -> CachedModel:
    """The model the endpoint currently serves (lineage, not name)."""
    return CachedModel(Endpoint(endpoint_name).get_input())


def _challengers(endpoint_name: str) -> list:
    """The models the endpoint's promotion node currently lists."""
    return [CachedModel(name) for name in CachedMeta().challenger_models(endpoint_name)]


@pytest.fixture(scope="module")
def regression_contest():
    """(champion, challengers) for the regression endpoint."""
    return _champion(REGRESSION_ENDPOINT), _challengers(REGRESSION_ENDPOINT)


@pytest.fixture(scope="module")
def classification_contest():
    """(champion, challengers) for the classification endpoint."""
    return _champion(CLASSIFICATION_ENDPOINT), _challengers(CLASSIFICATION_ENDPOINT)


def _contest(champ_value, challengers, metric="rmse"):
    """(champ_row, chall_rows) shaped like rank_models()/contest_ranking() output.

    challengers: [(name, value)] already ranked best-first, as contest_ranking() returns them.
    Δ is metrics-aware and absolute: champion - challenger for rmse (lower is better),
    challenger - champion for f1. Positive Δ always means the challenger is better.
    """
    champ_row = pd.DataFrame([{metric: champ_value}], index=["champ"])
    deltas = [(champ_value - v) if metric in ("rmse", "mae") else (v - champ_value) for _, v in challengers]
    chall_rows = pd.DataFrame(
        [{metric: v, f"Δ{metric}": d} for (_, v), d in zip(challengers, deltas)],
        index=[n for n, _ in challengers],
    )
    return champ_row, chall_rows


def test_regression_comparison(regression_contest):
    """Regressor comparison: [a, b, delta] rows with metrics-aware delta signs"""
    champion, challengers = regression_contest
    challenger = challengers[0]
    comp = model_comparison(champion, challenger, "full_cross_fold")
    assert list(comp.index) == [champion.name, challenger.name, "delta"]
    assert {"rmse", "mae", "r2", "spearmanr", "support"} <= set(comp.columns)

    # Metrics-aware: positive delta always means model_b is better
    row_a, row_b, delta = comp.iloc[0], comp.iloc[1], comp.loc["delta"]
    assert delta["rmse"] == pytest.approx(row_a["rmse"] - row_b["rmse"])  # lower is better
    assert delta["r2"] == pytest.approx(row_b["r2"] - row_a["r2"])  # higher is better


def test_classification_comparison(classification_contest):
    """Classifier comparison uses the 'all' summary row"""
    champion, challengers = classification_contest
    challenger = challengers[0]
    comp = model_comparison(champion, challenger, "full_cross_fold")
    assert list(comp.index) == [champion.name, challenger.name, "delta"]
    assert {"precision", "recall", "f1", "roc_auc"} <= set(comp.columns)


def test_missing_run_returns_none(regression_contest):
    """A missing inference run on either model returns None"""
    champion, challengers = regression_contest
    assert model_comparison(champion, challengers[0], "no-such-run") is None


def test_rank_models(regression_contest):
    """rank_models() sorts regressors by rmse (low to high)"""
    _, challengers = regression_contest
    ranked = rank_models(challengers, "full_cross_fold")
    assert list(ranked["rmse"]) == sorted(ranked["rmse"])


def test_contest_ranking(regression_contest):
    """contest_ranking() ranks challengers with metrics-aware Δ columns vs the champion"""
    champion, challengers = regression_contest
    ranked = contest_ranking(champion, challengers, "full_cross_fold")
    assert list(ranked.columns[:2]) == ["rmse", "Δrmse"]  # Δ interleaved after each metric
    assert "Δsupport" not in ranked.columns

    # Δrmse is metrics-aware: champion rmse minus challenger rmse (positive = challenger better)
    champ_rmse = champion.get_inference_metrics("full_cross_fold").iloc[0]["rmse"]
    for name, row in ranked.iterrows():
        assert row["Δrmse"] == pytest.approx(champ_rmse - row["rmse"])


def test_contest_report(regression_contest):
    """contest_report() has champion first (Δ=0), ranked challengers, and contest metadata columns"""
    champion, challengers = regression_contest
    report = contest_report(champion, challengers, REGRESSION_ENDPOINT, "full_cross_fold")

    # One champion row first; a challenger row for each that had metrics to rank
    assert report["role"].iloc[0] == "champion"
    assert set(report["role"].iloc[1:]) <= {"challenger"}
    assert report["model"].iloc[0] == champion.name
    assert report["endpoint"].eq(REGRESSION_ENDPOINT).all()
    assert report["framework"].isin(["xgboost", "pytorch", "chemprop", "chemprop-desc", "multi-task", "sklearn"]).all()
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


class _StubModel:
    """Just enough model to rank: a name, a metrics row, and a confusion matrix
    (None for a regressor, which has no matrix to collapse)."""

    METRICS = {"precision": 0.8, "recall": 0.79, "f1": 0.795, "roc_auc": 0.9, "support": 116}

    def __init__(self, conf_matrix, name="stub-model", metrics=None):
        self._conf_matrix = conf_matrix
        self.name = name
        self._metrics = metrics or self.METRICS

    def get_inference_metrics(self, inference_run):
        return pd.DataFrame([self._metrics])

    def confusion_matrix(self, inference_run):
        return self._conf_matrix


def _conf_matrix():
    # The standard form: a "labels" column (actual class) plus a column per predicted class
    return pd.DataFrame(
        {
            "labels": ["low", "med", "high"],
            "low": [40, 5, 1],
            "med": [6, 30, 4],
            "high": [2, 3, 25],
        }
    )


def test_binary_acc_collapses_to_desired_vs_undesired():
    """desired = low + med, so 'high' is the whole negative class"""
    acc = _binary_acc(_StubModel(_conf_matrix()), "full_cross_fold", ["low", "med"])
    tp, tn, fp, fn = 40 + 6 + 5 + 30, 25, 1 + 4, 2 + 3
    assert acc == pytest.approx((tp + tn) / (tp + tn + fp + fn))


@pytest.mark.parametrize(
    "conf_matrix, positive_classes",
    [
        (None, ["low"]),  # regressor (or no matrix for the run)
        (_conf_matrix(), ["low", "med", "high"]),  # every class desired -> accuracy is trivially 1
        (_conf_matrix(), ["not-a-class"]),  # none of the desired classes are present
    ],
)
def test_binary_acc_none_when_not_computable(conf_matrix, positive_classes):
    """A degenerate case yields None, so no column is added at all -- never a NaN column"""
    assert _binary_acc(_StubModel(conf_matrix), "full_cross_fold", positive_classes) is None


def test_rank_models_places_binary_acc_after_roc_auc():
    """Given desired classes, rank_models() slots binary_acc in right after roc_auc"""
    models = [_StubModel(_conf_matrix(), name="a"), _StubModel(_conf_matrix(), name="b")]
    ranked = rank_models(models, "full_cross_fold", positive_classes=["low", "med"])
    assert list(ranked.columns) == ["precision", "recall", "f1", "roc_auc", "binary_acc", "support"]

    # No desired classes, or none computable -> the table is exactly as it was
    assert "binary_acc" not in rank_models(models, "full_cross_fold").columns
    regressors = [_StubModel(None, name="a"), _StubModel(None, name="b")]
    assert "binary_acc" not in rank_models(regressors, "full_cross_fold", positive_classes=["low"]).columns


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


def test_prediction_comparison(regression_contest):
    """prediction_comparison() stacks both models' predictions with a 'model' column"""
    champion, challengers = regression_contest
    challenger = challengers[-1]
    preds = prediction_comparison(champion, challenger, "full_cross_fold")
    assert list(preds["model"].unique()) == [champion.name, challenger.name]
    assert {"prediction", "solubility"} <= set(preds.columns)


if __name__ == "__main__":
    regression = (_champion(REGRESSION_ENDPOINT), _challengers(REGRESSION_ENDPOINT))
    classification = (_champion(CLASSIFICATION_ENDPOINT), _challengers(CLASSIFICATION_ENDPOINT))
    test_regression_comparison(regression)
    test_classification_comparison(classification)
    test_missing_run_returns_none(regression)
    test_rank_models(regression)
    test_contest_ranking(regression)
    test_contest_report(regression)
    test_contested_skips_the_champions_twin()
    test_contested_percent_threshold()
    test_contested_classifier_and_edges()
    test_prediction_comparison(regression)
    print("All model_comparison tests passed!")
