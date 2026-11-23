"""Tests for the data comparison and data quality plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from workbench.utils.pandas_utils import compare_keyed, data_quality  # noqa: E402
from workbench.utils.plots.data import (  # noqa: E402
    one_sided_bars,
    outlier_box,
    overlap_venn,
    quality_list,
    quality_counts,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def compared():
    left = pd.DataFrame({"id": list("abcdeffg"), "route": ["po"] * 7 + ["ip"], "x": [1.0, 2, 3, 4, 5, 6, 6.5, 7]})
    right = pd.DataFrame({"id": list("abcdfh"), "route": ["po"] * 6, "x": [1.0, 2, 30, 4, 6, 8]})
    return left, right, compare_keyed(left, right, on="id", values=["x"], scope=["route"])


def venn_text(result, *names) -> list:
    return [t.get_text() for t in overlap_venn(result, *names).axes[0].texts]


def test_the_venn_counts_each_region_and_names_each_side():
    _, _, result = compared()
    text = venn_text(result, "archive", "DataSource")

    assert text[:4] == ["archive", "", "DataSource", ""]
    assert text[4:10] == ["1", "only here", "1", "only here", "3", "agree"]
    assert text[-2:] == ["1 mismatched, 1 repeated keys differ", "1 rows not expected, set aside"]


def test_the_venn_shows_where_a_side_lives_under_its_name():
    _, _, result = compared()
    long = "s3://a-bucket/" + "with/a/very/long/prefix/" * 4 + "archive"
    figure = overlap_venn(result, "Archive", "DataSource", locations={"Archive": long, "DataSource": "cmax_clean"})
    text = [t.get_text() for t in figure.axes[0].texts]

    assert text[0] == "Archive" and text[1].startswith("s3://a-bucket/") and text[1].endswith("/archive")
    assert len(text[1]) == 59 and "..." in text[1]
    assert text[2:4] == ["DataSource", "cmax_clean"]


def test_the_venn_says_nothing_about_differences_when_there_are_none():
    left = pd.DataFrame({"id": ["a", "a", "b"], "x": [1.0, 2.0, 3.0]})
    right = pd.DataFrame({"id": ["a", "a", "b"], "x": [2.0, 1.0, 3.0]})
    text = venn_text(compare_keyed(left, right, on="id"))

    assert "3" in text and not [t for t in text if "differ" in t or "mismatched" in t or "not expected" in t]


def test_one_sided_bars_count_each_sides_rows_per_group():
    left = pd.DataFrame({"id": list("abcd"), "year": ["2021", "2021", "2022", None]})
    right = pd.DataFrame({"id": list("aefg"), "year": ["2021", "2022", "2022", "2023"]})
    ax = one_sided_bars(compare_keyed(left, right, on="id"), "year", "Archive", "DataSource").axes[0]

    assert [t.get_text() for t in ax.get_yticklabels()] == ["(none)", "2021", "2022", "2023"]
    assert sum(patch.get_width() for patch in ax.patches) == 6
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["only in Archive", "only in DataSource"]


def test_one_sided_bars_fold_the_smallest_groups_into_other():
    left = pd.DataFrame({"id": range(40), "site": [f"s{i % 12}" for i in range(40)]})
    ax = one_sided_bars(compare_keyed(left, left.iloc[:0], on="id"), "site").axes[0]

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 8 and labels[-1] == "other"
    assert sum(patch.get_width() for patch in ax.patches) == 40


def test_one_sided_bars_refuse_a_column_neither_side_has():
    left = pd.DataFrame({"id": list("ab")})
    with pytest.raises(ValueError, match="year"):
        one_sided_bars(compare_keyed(left, left.iloc[:1], on="id"), "year")


def test_the_quality_list_ranks_columns_and_names_the_checks_behind_each_count():
    df = pd.DataFrame({"id": [f"c{i}" for i in range(40)], "a": np.arange(40.0), "b": np.arange(40.0)})
    df.loc[:9, "a"] = np.nan
    df.loc[:2, "b"] = np.nan
    df.loc[5, "b"] = np.inf
    text = [t.get_text() for t in quality_list(data_quality(df, key="id")).axes[0].texts]

    assert text == ["10", "a", "missing 10", "4", "b", "missing 3, non_finite 1"]


def test_the_quality_list_colours_a_warned_columns_count():
    df = pd.DataFrame({"id": [f"c{i}" for i in range(40)], "a": np.arange(40.0), "b": np.arange(40.0)})
    df.loc[0, "a"] = np.nan
    df.loc[[5, 6], "b"] = np.inf
    counts = {t.get_text(): t.get_color() for t in quality_list(data_quality(df, key="id")).axes[0].texts}

    assert counts["1"] != counts["2"]  # a is missing one value (minor), b holds infinities (a warning)


def test_the_quality_list_is_empty_for_a_clean_frame():
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "x": [1.0, 2.0, 3.0, 4.0]})
    assert not quality_list(data_quality(df, key="id")).axes[0].texts


def test_whole_row_findings_are_counted_under_their_own_label():
    df = pd.DataFrame({"id": list("aabbc"), "x": [1.0, 1.0, 2.0, 2.0, np.nan]})
    counts = quality_counts(data_quality(df)).sum(axis=1)
    assert counts.idxmax() == "(whole row)" and counts["(whole row)"] == 4 and counts["x"] == 1


def test_the_outlier_box_marks_each_value_outside_the_fence():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": range(400), "cmax": 10 ** rng.normal(2, 1, 400)})
    df.loc[[5, 9], "cmax"] = 0.0
    ax = outlier_box(data_quality(df, key="id"), "cmax").axes[0]

    (points,) = [c for c in ax.collections if len(c.get_offsets())]
    assert points.get_offsets()[:, 1].tolist() == [0.0, 0.0]
    assert ax.get_yscale() == "symlog"  # a zero has no place on a log axis
    assert ax.get_xlabel().startswith("2 outside\nfence ")


def test_the_outlier_box_refuses_a_column_without_outliers():
    with pytest.raises(ValueError, match="no outliers"):
        outlier_box(data_quality(pd.DataFrame({"x": [1.0, 2.0, 3.0]})), "x")
