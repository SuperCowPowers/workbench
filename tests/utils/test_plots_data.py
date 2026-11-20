"""Tests for the data comparison and data quality plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

from workbench.utils.pandas_utils import compare_keyed, data_quality  # noqa: E402
from workbench.utils.plots.data import (  # noqa: E402
    comparison_fates,
    fate_bars,
    quality_bars,
    quality_counts,
    value_agreement,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def compared():
    left = pd.DataFrame({"id": list("abcdeffg"), "route": ["po"] * 7 + ["ip"], "x": [1.0, 2, 3, 4, 5, 6, 6, 7]})
    right = pd.DataFrame({"id": list("abcdfh"), "route": ["po"] * 6, "x": [1.0, 2, 30, 4, 6, 8]})
    return left, right, compare_keyed(left, right, on="id", values=["x"], scope=["route"])


def test_fates_account_for_every_row_of_each_side():
    left, right, result = compared()
    fates = comparison_fates(result)

    assert fates["left"].sum() == len(left) and fates["right"].sum() == len(right)
    assert fates.loc["agree"].tolist() == [3, 3] and fates.loc["mismatched"].tolist() == [1, 1]
    assert fates.loc["ambiguous"].tolist() == [2, 1]
    assert fates.loc["one side only"].tolist() == [1, 1] and fates.loc["out of scope"].tolist() == [1, 0]


def test_fate_bars_stack_to_each_sides_row_count():
    left, right, result = compared()
    ax = fate_bars(result, "archive", "DataSource").axes[0]
    widths = {1: 0, 0: 0}
    for patch in ax.patches:
        widths[round(patch.get_y() + patch.get_height() / 2)] += patch.get_width()

    assert widths == {1: len(left), 0: len(right)}
    assert [t.get_text() for t in ax.get_yticklabels()] == ["archive", "DataSource"]


def test_value_agreement_draws_every_pair_and_marks_the_mismatches():
    _, _, result = compared()
    ax = value_agreement(result, "x", "archive", "DataSource").axes[0]
    agree, off = [c for c in ax.collections if isinstance(c, PathCollection)]

    assert len(agree.get_offsets()) == 3 and len(off.get_offsets()) == 1
    assert off.get_offsets()[0].tolist() == [30.0, 3.0]
    assert ax.get_xlabel() == "x in DataSource" and ax.get_ylabel() == "x in archive"


def test_quality_bars_rank_columns_by_flagged_rows():
    df = pd.DataFrame({"id": [f"c{i}" for i in range(40)], "a": np.arange(40.0), "b": np.arange(40.0)})
    df.loc[:9, "a"] = np.nan
    df.loc[:2, "b"] = np.nan
    df.loc[5, "b"] = np.inf
    ax = quality_bars(data_quality(df, key="id")).axes[0]

    ticks = dict(zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()))
    assert ticks["a"] > ticks["b"]
    assert sum(patch.get_width() for patch in ax.patches) == 14


def test_quality_bars_draw_nothing_for_a_clean_frame():
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "x": [1.0, 2.0, 3.0, 4.0]})
    assert len(quality_bars(data_quality(df, key="id")).axes[0].patches) == 0


def test_whole_row_findings_are_counted_under_their_own_label():
    df = pd.DataFrame({"id": list("aabbc"), "x": [1.0, 1.0, 2.0, 2.0, np.nan]})
    counts = quality_counts(data_quality(df)).sum(axis=1)
    assert counts.idxmax() == "(whole row)" and counts["(whole row)"] == 4 and counts["x"] == 1


def test_an_empty_side_draws_no_zero_labels():
    left = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
    ax = fate_bars(compare_keyed(left, left.iloc[:0], on="id")).axes[0]
    assert [t.get_text() for t in ax.texts] == ["2"]
