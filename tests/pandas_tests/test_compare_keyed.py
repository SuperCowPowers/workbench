"""Tests for compare_keyed: key overlap, value agreement, and scoring only one-to-one keys"""

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.utils.pandas_utils import compare_keyed


def test_overlap_buckets():
    """Every row lands in exactly one bucket"""
    left = pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    right = pd.DataFrame({"id": ["b", "c", "d"], "x": [2.0, 9.0, 4.0]})
    result = compare_keyed(left, right, on="id")

    assert list(result.pairs["id"]) == ["b", "c"]
    assert list(result.left_only["id"]) == ["a"]
    assert list(result.right_only["id"]) == ["d"]
    assert result.repeated.empty
    assert list(result.mismatched["id"]) == ["c"]
    assert result.summary()["agree"] == 1


def test_pairs_carry_both_values_and_indexes():
    """A mismatch is actionable: both values and each side's original index label"""
    left = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]}, index=[10, 11])
    right = pd.DataFrame({"id": ["b", "a"], "x": [5.0, 1.0]}, index=[20, 21])
    row = compare_keyed(left, right, on="id").mismatched.iloc[0]

    assert (row["id"], row["left_index"], row["right_index"]) == ("b", 11, 20)
    assert (row["x_left"], row["x_right"], row["x_agree"]) == (2.0, 5.0, False)


def test_repeated_keys_are_judged_as_a_set():
    """A key repeated on either side is not a pile of disagreements: its rows are compared as a set"""
    left = pd.DataFrame({"id": ["a", "a", "b", "c"], "x": [1.0, 2.0, 3.0, 4.0]})
    right = pd.DataFrame({"id": ["a", "a", "b", "c", "c"], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})

    # The naive join scores 4 "a" rows and 2 "c" rows, 3 of them as disagreements
    naive = left.merge(right, on="id", suffixes=("_left", "_right"))
    assert (naive["x_left"] != naive["x_right"]).sum() == 3

    result = compare_keyed(left, right, on="id")
    assert list(result.pairs["id"]) == ["b"]
    assert result.mismatched.empty
    assert result.repeated.to_dict("records") == [
        {"id": "a", "left_rows": 2, "right_rows": 2, "values_match": True, "x_left": "1; 2", "x_right": "1; 2"},
        {"id": "c", "left_rows": 1, "right_rows": 2, "values_match": False, "x_left": "4", "x_right": "4; 5"},
    ]
    assert list(result.repeated_differ["id"]) == ["c"]
    assert result.left_only.empty and result.right_only.empty
    # "b" pairs one to one and the two "a" rows match as a set, so three pairs agree
    assert result.summary()[["pairs", "agree", "mismatched", "repeated_differ"]].tolist() == [3, 3, 0, 1]


def test_repeated_keys_match_whatever_order_the_rows_come_in():
    """Same values in a different row order agree; one different value does not"""
    left = pd.DataFrame({"id": ["a"] * 3, "x": [3.0, 1.0, 2.0], "s": ["z", "x", "y"]})
    right = pd.DataFrame({"id": ["a"] * 3, "x": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]})
    assert compare_keyed(left, right, on="id").repeated["values_match"].tolist() == [True]
    assert compare_keyed(left, right.assign(x=[1.0, 2.0, 3.5]), on="id").repeated["values_match"].tolist() == [False]
    assert compare_keyed(left, right.assign(x=[1.0, 2.0, 3.05]), on="id", rtol=0.02).repeated["values_match"].all()


def test_repeated_keys_match_on_their_values_not_their_row_counts():
    """Three rows of 1.95 against one row of 1.95 is the same result recorded more often"""
    left = pd.DataFrame({"id": ["a"] * 3 + ["b"] * 2 + ["c"] * 2, "x": [1.95, 1.95, 1.95, 185.0, 185.0, 4.0, 5.0]})
    right = pd.DataFrame({"id": ["a", "b", "c"], "x": [1.95, 185.0, 4.0]})
    result = compare_keyed(left, right, on="id", repeats="values")

    assert result.repeated.set_index("id")["values_match"].to_dict() == {"a": True, "b": True, "c": False}
    assert list(result.repeated_differ["id"]) == ["c"]  # the left holds a 5 the right does not
    assert result.summary()[["agree", "repeated_differ"]].tolist() == [5, 1]

    # By default the rows themselves must match, so a different row count is a difference
    assert list(compare_keyed(left, right, on="id").repeated_differ["id"]) == ["a", "b", "c"]


def test_exact_repeats_compare_the_rows_not_just_the_distinct_values():
    left = pd.DataFrame({"id": ["a"] * 3, "x": [1.0, 1.0, 2.0]})
    right = pd.DataFrame({"id": ["a"] * 3, "x": [1.0, 2.0, 2.0]})
    assert not compare_keyed(left, right, on="id").repeated["values_match"].any()
    assert compare_keyed(left, right, on="id", repeats="values").repeated["values_match"].all()
    with pytest.raises(ValueError, match="repeats"):
        compare_keyed(left, right, on="id", repeats="loose")


def test_the_result_records_how_the_comparison_was_made():
    left = pd.DataFrame({"id": [1, 2], "route": ["po", "ip"], "x": [1.0, 2.0]})
    right = pd.DataFrame({"id": [1], "route": ["po"], "x": [1.0]})
    result = compare_keyed(left, right, on="id", rtol=0.02, scope=["route"])
    assert (result.values, result.scope, result.rtol, result.repeats) == (["x"], {"route": ["po"]}, 0.02, "exact")


def test_repeated_key_on_one_side_only_stays_one_sided():
    """A repeated key the other side lacks is one-sided rows, not a repeated key to judge"""
    left = pd.DataFrame({"id": ["a", "a"], "x": [1.0, 2.0]})
    right = pd.DataFrame({"id": ["b"], "x": [1.0]})
    result = compare_keyed(left, right, on="id")

    assert len(result.left_only) == 2
    assert result.repeated.empty


def test_tolerance():
    """rtol is a fraction of the right value; atol is absolute"""
    left = pd.DataFrame({"id": [1, 2, 3], "x": [101.0, 103.0, 0.001]})
    right = pd.DataFrame({"id": [1, 2, 3], "x": [100.0, 100.0, 0.0]})

    assert list(compare_keyed(left, right, on="id").pairs["agree"]) == [False, False, False]
    assert list(compare_keyed(left, right, on="id", rtol=0.02).pairs["agree"]) == [True, False, False]
    assert list(compare_keyed(left, right, on="id", rtol=0.02, atol=0.01).pairs["agree"]) == [True, False, True]


def test_nan_and_string_values():
    """NaN agrees with NaN; strings compare by equality"""
    left = pd.DataFrame({"id": [1, 2, 3], "x": [np.nan, 1.0, np.nan], "s": ["po", "iv", None]})
    right = pd.DataFrame({"id": [1, 2, 3], "x": [np.nan, np.nan, 2.0], "s": ["po", "ip", None]})
    pairs = compare_keyed(left, right, on="id").pairs

    assert list(pairs["x_agree"]) == [True, False, False]
    assert list(pairs["s_agree"]) == [True, False, True]
    assert list(pairs["agree"]) == [True, False, False]


def test_nan_key_matches_nothing():
    """A row with a NaN in any key column is one-sided, even against another NaN"""
    left = pd.DataFrame({"id": ["a", None], "dose": [1.0, 1.0], "x": [1, 2]})
    right = pd.DataFrame({"id": ["a", None], "dose": [1.0, 1.0], "x": [1, 2]})
    result = compare_keyed(left, right, on=["id", "dose"])

    assert len(result.pairs) == 1
    assert len(result.left_only) == 1 and len(result.right_only) == 1


def test_scope():
    """A row holding a scope value the other side never carries is out of scope, not missing"""
    archive = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "route": ["po", "iv", "ip", "po"],
            "matrix": ["plasma", "plasma", "plasma", "blood"],
            "cmax": [1.0, 2.0, 3.0, 4.0],
        }
    )
    source = pd.DataFrame({"id": ["a"], "route": ["po"], "matrix": ["plasma"], "cmax": [1.0]})
    source = pd.concat([source, pd.DataFrame({"id": ["z"], "route": ["iv"], "matrix": ["plasma"], "cmax": [9.0]})])

    unscoped = compare_keyed(archive, source, on="id", values=["cmax"])
    assert len(unscoped.left_only) == 3

    scoped = compare_keyed(archive, source, on="id", values=["cmax"], scope=["route", "matrix"])
    assert list(scoped.left_only["id"]) == ["b"]
    assert list(scoped.left_out_of_scope["id"]) == ["c", "d"]
    assert list(scoped.left_out_of_scope["scope_reason"]) == ["route = ip", "matrix = blood"]
    assert scoped.right_out_of_scope.empty
    assert list(scoped.right_only["id"]) == ["z"]


def test_default_values_exclude_keys_and_scope():
    """With no `values`, every shared column outside `on` and `scope` is compared"""
    left = pd.DataFrame({"id": [1], "route": ["po"], "x": [1.0], "left_extra": [0]})
    right = pd.DataFrame({"id": [1], "route": ["po"], "x": [1.0], "right_extra": [0]})
    pairs = compare_keyed(left, right, on="id", scope=["route"]).pairs

    assert list(pairs.columns) == ["id", "left_index", "right_index", "x_left", "x_right", "x_agree", "agree"]


def test_no_values_reports_overlap_only():
    """Frames sharing only the key still report overlap, and every pair agrees"""
    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"id": [2, 3]})
    result = compare_keyed(left, right, on="id")

    assert list(result.pairs["id"]) == [2]
    assert result.pairs["agree"].all()


def test_missing_column_raises():
    left = pd.DataFrame({"id": [1], "x": [1.0]})
    right = pd.DataFrame({"id": [1]})
    with pytest.raises(ValueError, match="right"):
        compare_keyed(left, right, on="id", values=["x"])


def test_zero_tolerance_compares_integers_exactly():
    """Integers past float precision still differ when no tolerance is asked for"""
    left = pd.DataFrame({"id": [1, 2], "n": [9007199254740992, 5]})
    right = pd.DataFrame({"id": [1, 2], "n": [9007199254740993, 5]})

    assert list(compare_keyed(left, right, on="id").pairs["agree"]) == [False, True]
    assert list(compare_keyed(left, right, on="id", atol=2).pairs["agree"]) == [True, True]


def test_scope_is_split_by_position_when_index_labels_repeat():
    left = pd.DataFrame({"id": ["a", "b", "c"], "route": ["po", "ip", "po"]}, index=[0, 0, 1])
    right = pd.DataFrame({"id": ["a", "c"], "route": ["po", "po"]})
    result = compare_keyed(left, right, on="id", scope=["route"])

    assert list(result.left_out_of_scope["id"]) == ["b"]
    assert list(result.pairs["id"]) == ["a", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
