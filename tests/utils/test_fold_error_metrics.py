"""Tests for the fold-error metrics in metrics_utils."""

import numpy as np
import pytest

import pandas as pd

from workbench.utils.metrics_utils import afe, fold_summary, gmfe, performance_breakout, within_fold

TRUE = np.array([1.0, 10.0, 100.0, 4.0])
PRED = np.array([2.0, 5.0, 100.0, 16.0])  # 2x high, 2x low, exact, 4x high


def test_gmfe_is_the_geometric_mean_of_the_fold_misses():
    assert gmfe(TRUE, PRED) == pytest.approx((2 * 2 * 1 * 4) ** 0.25)
    assert gmfe(TRUE, TRUE) == pytest.approx(1.0)


def test_afe_keeps_the_sign_that_gmfe_drops():
    assert afe(TRUE, PRED) == pytest.approx((2 * 0.5 * 1 * 4) ** 0.25)
    assert afe(TRUE, TRUE * 3) == pytest.approx(3.0)
    assert afe(TRUE, TRUE / 3) == pytest.approx(1 / 3)


def test_within_fold_counts_both_directions_and_the_boundary():
    assert within_fold(TRUE, PRED) == pytest.approx(0.75)
    assert within_fold(TRUE, PRED, fold=4.0) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [gmfe, afe, within_fold])
def test_nonpositive_values_raise(metric):
    with pytest.raises(ValueError):
        metric(TRUE, np.array([2.0, 0.0, 100.0, 16.0]))


def test_fold_summary_matches_the_individual_metrics():
    summary = fold_summary(TRUE, PRED)
    assert summary["gmfe"] == pytest.approx(gmfe(TRUE, PRED))
    assert summary["afe"] == pytest.approx(afe(TRUE, PRED))
    assert summary["within_2x"] == pytest.approx(0.75)
    assert summary["within_3x"] == pytest.approx(0.75)
    assert summary["within_10x"] == pytest.approx(1.0)
    assert summary["support"] == 4


def test_fold_summary_drops_nan_and_nonpositive_pairs():
    true = np.append(TRUE, [np.nan, 0.0, 5.0, 5.0])
    pred = np.append(PRED, [1.0, 1.0, -1.0, np.nan])
    assert fold_summary(true, pred) == fold_summary(TRUE, PRED)
    assert fold_summary([0.0, np.nan], [1.0, 1.0]) == {}


def test_performance_breakout_scores_all_rows_then_each_group():
    df = pd.DataFrame({"y": list(TRUE) * 3, "prediction": list(PRED) * 3, "project": ["a"] * 6 + ["b"] * 5 + ["c"]})
    table = performance_breakout(df, "y", by="project", log=True, min_support=5)

    assert list(table["group"]) == ["all", "a", "b", "other"]
    assert list(table["support"]) == [12, 6, 5, 1]
    assert table.loc[0, "gmfe"] == pytest.approx(gmfe(TRUE, PRED))
    assert table.loc[0, "rmse"] == pytest.approx(np.sqrt(np.mean(np.log10(PRED / TRUE) ** 2)))


def test_performance_breakout_without_by_is_one_row_in_target_units():
    df = pd.DataFrame({"y": TRUE, "prediction": PRED})
    table = performance_breakout(df, "y")
    assert list(table["group"]) == ["all"] and "gmfe" not in table
    assert table.loc[0, "mae"] == pytest.approx(np.mean(np.abs(PRED - TRUE)))


def test_performance_breakout_keeps_its_columns_when_nothing_can_be_scored():
    df = pd.DataFrame({"y": [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], "prediction": [1.0, 1.1, 2.1, 2.9, 4.2, 5.1]})
    df["project"] = ["a"] * 1 + ["b"] * 5
    table = performance_breakout(df.assign(y=np.nan), "y", by="project", log=True, min_support=1)

    assert list(table["support"]) == [0, 0, 0]
    assert table["gmfe"].isna().all() and "within_10x" in table
    assert list(performance_breakout(df, "y", by="project", min_support=1)["support"]) == [5, 5, 0]
