"""Tests for the model performance plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from workbench.utils.plots.performance import breakout_bars, fold_bands, pred_vs_actual, residuals  # noqa: E402

RNG = np.random.default_rng(0)
TRUE = 10 ** RNG.uniform(0, 3, 200)
PRED = TRUE * 10 ** RNG.normal(0, 0.2, 200)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def points(ax) -> int:
    return sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection))


def test_pred_vs_actual_linear_is_square_and_drops_nans():
    true = np.append(TRUE, [np.nan, 5.0])
    pred = np.append(PRED, [5.0, np.nan])
    ax = pred_vs_actual(true, pred, unit="nM").axes[0]

    assert points(ax) == 200
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_xlabel() == "Measured (nM)" and ax.get_ylabel() == "Predicted (nM)"


def test_pred_vs_actual_log_uses_decade_limits_and_leaves_off_nonpositives():
    true = np.append(TRUE, [0.0, -1.0])
    pred = np.append(PRED, [1.0, 1.0])
    ax = pred_vs_actual(true, pred, log=True).axes[0]

    assert points(ax) == 200
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    lo, hi = ax.get_xlim()
    assert np.log10(lo) == pytest.approx(round(np.log10(lo))) and np.log10(hi) == pytest.approx(round(np.log10(hi)))
    assert lo <= min(TRUE.min(), PRED.min()) and hi >= max(TRUE.max(), PRED.max())


def test_pred_vs_actual_groups_get_a_legend_and_pool_the_tail():
    groups = np.array([f"project_{i % 8}" for i in range(200)])
    ax = pred_vs_actual(TRUE, PRED, groups=groups, log=True).axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]

    assert len(labels) == 7 and labels[-1] == "other"
    assert points(ax) == 200


def test_pred_vs_actual_draws_on_a_given_axes():
    fig, ax = plt.subplots()
    assert pred_vs_actual(TRUE, PRED, ax=ax) is fig
    assert len(fig.axes) == 1


def test_large_scatters_are_rasterized():
    big = RNG.uniform(1, 10, 2500)
    assert pred_vs_actual(big, big).axes[0].collections[-1].get_rasterized()
    assert not pred_vs_actual(TRUE, PRED).axes[0].collections[-1].get_rasterized()


def test_fold_bands_sets_equal_log_axes():
    _, ax = plt.subplots()
    fold_bands(ax, 1.0, 1000.0)
    assert ax.get_xlim() == (1.0, 1000.0) and ax.get_ylim() == (1.0, 1000.0)
    assert len(ax.lines) == 5  # 1:1, and 3x and 10x each side


def test_residuals_counts_every_pair():
    ax = residuals(TRUE, PRED, unit="nM").axes[0]
    assert sum(p.get_height() for p in ax.patches) == 200
    assert ax.get_xlabel() == "Predicted - measured (nM)"
    assert residuals(TRUE, PRED, log=True).axes[0].get_xlabel() == "log10(predicted / measured)"


def test_breakout_bars_keep_the_given_order_top_to_bottom():
    values = pd.Series({"alpha": 0.9, "beta": 0.5, "gamma": 0.7})
    ax = breakout_bars(values, reference=0.6, xlabel="R2").axes[0]

    ticks = dict(zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()))
    assert ticks["alpha"] > ticks["beta"] > ticks["gamma"]
    assert [p.get_width() for p in ax.patches] == [0.9, 0.5, 0.7]
    assert ax.lines[0].get_xdata()[0] == 0.6


def test_infinite_pairs_are_left_off():
    true = np.append(TRUE, [np.inf, 5.0])
    pred = np.append(PRED, [5.0, -np.inf])

    assert points(pred_vs_actual(true, pred).axes[0]) == 200
    assert sum(p.get_height() for p in residuals(true, pred).axes[0].patches) == 200


def test_an_input_group_named_other_is_drawn_once():
    groups = np.array(["other"] * 150 + ["alpha"] * 50)
    ax = pred_vs_actual(TRUE, PRED, groups=groups).axes[0]

    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["alpha", "other"]
    assert points(ax) == 200
