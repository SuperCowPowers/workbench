"""Tests for data_quality: each check, and the rows behind every finding"""

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.utils.pandas_utils import data_quality, noise_floor


def finding(result, check: str, column: str = None) -> pd.Series:
    """The single finding for a check (and column)."""
    hits = result.findings[result.findings["check"] == check]
    if column is not None:
        hits = hits[hits["column"] == column]
    assert len(hits) == 1, f"expected one {check} finding, got {len(hits)}"
    return hits.iloc[0]


def rows(result, check: str) -> pd.DataFrame:
    return result.rows[result.rows["check"] == check]


def clean_frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"id": [f"c{i}" for i in range(n)], "x": rng.normal(size=n), "y": rng.normal(size=n)})


def test_clean_frame_has_no_findings():
    result = data_quality(clean_frame(), key="id")
    assert result.findings.empty
    assert result.rows.empty
    assert list(result.rows.columns) == ["id", "check", "column", "value"]


def test_missing_lists_the_rows_by_key():
    df = clean_frame()
    df.loc[[3, 7], "x"] = np.nan
    result = data_quality(df, key="id")

    assert finding(result, "missing", "x")["severity"] == "info"
    assert finding(result, "missing", "x")["value"] == "2 (4.0%)"
    assert list(rows(result, "missing")["id"]) == ["c3", "c7"]


def test_missing_is_a_warning_when_widespread_or_in_the_key():
    df = clean_frame()
    df.loc[:9, "x"] = np.nan
    df.loc[20, "id"] = None
    result = data_quality(df, key="id")

    assert finding(result, "missing", "x")["severity"] == "warn"
    assert finding(result, "missing", "id")["severity"] == "warn"


def test_non_finite_carries_the_value():
    df = clean_frame()
    df.loc[4, "x"] = np.inf
    df.loc[5, "x"] = -np.inf
    result = data_quality(df, key="id")

    assert finding(result, "non_finite", "x")["severity"] == "warn"
    assert rows(result, "non_finite")[["id", "value"]].values.tolist() == [["c4", "inf"], ["c5", "-inf"]]


def test_outliers_ignore_infinities_and_report_the_fence():
    df = clean_frame()
    df.loc[1, "x"] = 500.0
    df.loc[2, "x"] = np.inf
    result = data_quality(df, key="id")

    assert list(rows(result, "outliers")["id"]) == ["c1"]
    assert "3 x IQR" in finding(result, "outliers", "x")["detail"]


def test_unparseable_text_in_a_numeric_column():
    df = clean_frame(10)
    df["conc"] = ["1.2", "3.4", "<0.1", "5.6", "BLQ", "7.8", "9.0", "1.1", "2.2", "3.3"]
    df["route"] = ["po", "iv"] * 5
    result = data_quality(df, key="id")

    assert rows(result, "unparseable")[["id", "value"]].values.tolist() == [["c2", "<0.1"], ["c4", "BLQ"]]
    assert result.findings[result.findings["column"] == "route"].empty


def test_constant_column():
    df = clean_frame().assign(species="mouse")
    result = data_quality(df, key="id")
    assert finding(result, "constant", "species")["value"] == "mouse"


def test_duplicate_key_and_duplicate_rows():
    df = clean_frame()
    df = pd.concat([df, df.iloc[[0]], df.iloc[[1]].assign(x=99.0)], ignore_index=True)
    result = data_quality(df, key="id", outlier_scale=1e9)

    assert sorted(rows(result, "duplicate_key")["id"]) == ["c0", "c0", "c1", "c1"]
    assert list(rows(result, "duplicate_rows")["id"]) == ["c0", "c0"]


def test_without_a_key_rows_are_listed_by_index():
    df = clean_frame().drop(columns="id")
    df.loc[6, "x"] = np.nan
    result = data_quality(df)
    assert list(rows(result, "missing")["index"]) == [6]


def test_multi_column_key():
    df = clean_frame().assign(dose=1.0)
    df.loc[2, "x"] = np.nan
    result = data_quality(df, key=["id", "dose"])
    assert rows(result, "missing")[["id", "dose"]].values.tolist() == [["c2", 1.0]]


def test_max_rows_caps_the_listing_not_the_count():
    df = clean_frame()
    df.loc[:19, "x"] = np.nan
    result = data_quality(df, key="id", max_rows=5)

    assert finding(result, "missing", "x")["value"] == "20 (40.0%)"
    assert "first 5 rows listed" in finding(result, "missing", "x")["detail"]
    assert len(rows(result, "missing")) == 5


def test_target_health_is_appended_and_findings_sort_worst_first():
    df = clean_frame(100)
    df["target"] = list(np.linspace(0, 90, 80)) + [100.0] * 20
    df.loc[0, "x"] = np.nan
    result = data_quality(df, key="id", target="target")

    assert finding(result, "censoring", "target")["severity"] == "warn"
    assert (result.findings["check"] == "missing").sum() == 1
    severities = list(result.findings["severity"])
    assert severities == sorted(severities, key=["warn", "info", "ok"].index)


def test_unhashable_columns_are_skipped():
    df = clean_frame(5).assign(fp=[[1, 0]] * 5)
    result = data_quality(df, key="id")
    assert result.findings.empty


def test_repeated_index_labels_do_not_break_the_checks():
    """A concatenated frame repeats index labels; rows are still found by position"""
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "conc": ["1", "BLQ", None, "2.5"]}, index=[0, 0, 1, 1])
    result = data_quality(df, key="id")

    assert list(rows(result, "unparseable")["id"]) == ["b"]
    assert list(rows(result, "missing")["id"]) == ["c"]


def test_rows_differing_only_in_an_uncomparable_column_are_not_duplicates():
    df = pd.DataFrame({"x": [1, 1], "fp": [[1, 0], [0, 1]]})
    assert "duplicate_rows" not in set(data_quality(df).findings["check"])


def test_a_log_distributed_column_is_fenced_on_its_log_values():
    """The upper tail of a concentration is not a pile of outliers, but a value decades away is"""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": [f"c{i}" for i in range(400)], "cmax": 10 ** rng.normal(2, 1, 400)})
    assert "outliers" not in set(data_quality(df, key="id").findings["check"])

    df.loc[0, "cmax"] = 1e12
    result = data_quality(df, key="id")
    assert list(rows(result, "outliers")["id"]) == ["c0"]
    assert "log10" in finding(result, "outliers", "cmax")["detail"]


def test_findings_carry_a_row_count():
    df = clean_frame().assign(species="mouse")
    df.loc[[3, 7], "x"] = np.nan
    findings = data_quality(df, key="id").findings.set_index("check")

    assert list(findings.columns) == ["column", "severity", "count", "value", "detail"]
    assert findings.loc["missing", "count"] == 2
    assert pd.isna(findings.loc["constant", "count"])


def test_a_zero_does_not_cost_a_log_distributed_column_its_log_fence():
    """Zeros in a concentration are the outliers, not its whole upper tail"""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": [f"c{i}" for i in range(400)], "cmax": 10 ** rng.normal(2, 1, 400)})
    df.loc[[5, 9], "cmax"] = 0.0
    result = data_quality(df, key="id")

    assert list(rows(result, "outliers")["id"]) == ["c5", "c9"]
    assert "log10" in finding(result, "outliers", "cmax")["detail"]


def test_duplicate_rows_carry_their_fields_with_copies_side_by_side():
    df = pd.DataFrame({"route": ["po", "iv", "po", "iv", "sc"], "dose": [10, 3, 10, 3, 1]})
    listed = rows(data_quality(df), "duplicate_rows")

    assert list(listed["value"]) == ["route=iv, dose=3", "route=iv, dose=3", "route=po, dose=10", "route=po, dose=10"]
    assert list(listed["index"]) == [1, 3, 0, 2]


def test_a_duplicate_key_lists_what_else_each_row_holds_grouped_by_key():
    df = pd.DataFrame({"id": ["b", "a", "b", "a"], "x": [1, 2, 3, 4]})
    listed = rows(data_quality(df, key="id"), "duplicate_key")

    assert listed[["id", "value"]].values.tolist() == [["a", "x=2"], ["a", "x=4"], ["b", "x=1"], ["b", "x=3"]]


def test_outliers_keep_what_a_box_plot_needs():
    df = clean_frame(200)
    df.loc[1, "x"] = 500.0
    box = data_quality(df, key="id").boxes.set_index("column").loc["x"]

    assert box["q1"] < box["median"] < box["q3"] and box["whisker_high"] < box["fence_high"] < 500
    assert box["whisker_low"] >= box["fence_low"] and not box["log"]
    assert data_quality(clean_frame(), key="id").boxes.empty


def test_noise_floor_pools_the_spread_of_repeated_keys():
    df = pd.DataFrame({"cpd": ["a", "a", "b", "b", "c"], "y": [1.0, 3.0, 10.0, 14.0, 7.0]})
    noise = noise_floor(df, "y", "cpd", log_scale=False)

    assert noise.groups == 2 and noise.rows == 4 and not noise.log and noise.rough
    assert noise.sd == pytest.approx(np.sqrt((2 + 8) / 2))  # squared deviations 1+1 and 4+4, two degrees of freedom
    assert noise.replicates[["cpd", "n"]].values.tolist() == [["a", 2], ["b", 2]]
    assert noise.max_r2 == pytest.approx(1 - noise.sd**2 / df["y"].var())
    assert noise.fold is None


def test_noise_floor_goes_log_on_a_skewed_positive_target_and_ignores_exact_copies():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cpd": np.repeat([f"c{i}" for i in range(40)], 3)})
    df["y"] = np.repeat(10 ** rng.uniform(0, 5, 40), 3) * 10 ** rng.normal(0, 0.1, 120)
    noise = noise_floor(pd.concat([df, df.head(30)]), "y", "cpd")

    assert noise.log and noise.groups == 40 and noise.rows == 120 and not noise.rough
    assert 0.07 < noise.sd < 0.13 and noise.fold == pytest.approx(10**noise.sd)


def test_data_quality_estimates_noise_by_replicate_key_or_key():
    df = pd.DataFrame({"row": range(4), "cpd": ["a", "a", "b", "b"], "y": [1.0, 2.0, 5.0, 5.5]})

    assert data_quality(df, key="row", target="y", replicate_key="cpd").noise.groups == 2
    assert data_quality(df, key="row", target="y").noise.groups == 0
    assert data_quality(df, key="row").noise is None
    assert data_quality(df, key="row", target="y", replicate_key="cpd", log_scale=True).noise.log


def test_noise_floor_ceiling_rises_for_a_mean_of_repeats_and_is_withheld_when_noise_exceeds_the_spread():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cpd": np.repeat(np.arange(200), 2)})
    df["y"] = np.repeat(rng.normal(0, 2, 200), 2) + rng.normal(0, 1, 400)
    noise = noise_floor(df, "y", "cpd", log_scale=False)

    assert noise.max_r2 == pytest.approx(0.8, abs=0.05)  # between 4, noise 1
    assert noise.max_r2_of_mean(2) == pytest.approx(4 / 4.5, abs=0.04) and noise.typical_repeats == 2
    low, high = noise.sd_interval()
    assert low < noise.sd < high and low < 1 < high

    # Repeats only among wild keys: noisier than the frame as a whole
    wild = pd.DataFrame({"cpd": ["a", "a", "b", "b"] + list("cdefghij"), "y": [0, 90, 10, 80] + [50.0] * 8})
    noise = noise_floor(wild, "y", "cpd", log_scale=False)
    assert noise.exceeds_spread and np.isnan(noise.max_r2) and np.isnan(noise.max_r2_of_mean(3))


def test_noise_floor_says_when_a_few_keys_carry_it_and_when_spread_moves_with_the_value():
    rng = np.random.default_rng(1)
    means = np.repeat(rng.uniform(1, 100, 100), 3)
    df = pd.DataFrame({"cpd": np.repeat(np.arange(100), 3), "y": means * (1 + rng.normal(0, 0.2, 300))})
    proportional = noise_floor(df, "y", "cpd", log_scale=False)
    assert proportional.trend > 0.5 and abs(noise_floor(df, "y", "cpd", log_scale=True).trend) < 0.3

    df = pd.DataFrame({"cpd": np.repeat(np.arange(100), 2), "y": rng.normal(0, 1, 200)})
    assert not noise_floor(df, "y", "cpd", log_scale=False).concentrated
    df.loc[0, "y"] = 100.0
    spiked = noise_floor(df, "y", "cpd", log_scale=False)
    assert spiked.concentrated and spiked.top_groups == 5 and spiked.top_share > 0.9
    assert spiked.median_sd < 1.5 < spiked.sd


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
