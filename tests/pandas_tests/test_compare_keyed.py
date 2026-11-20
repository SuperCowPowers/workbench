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
    assert result.ambiguous.empty
    assert list(result.mismatched["id"]) == ["c"]
    assert result.summary()["agree"] == 1


def test_pairs_carry_both_values_and_indexes():
    """A mismatch is actionable: both values and each side's original index label"""
    left = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]}, index=[10, 11])
    right = pd.DataFrame({"id": ["b", "a"], "x": [5.0, 1.0]}, index=[20, 21])
    row = compare_keyed(left, right, on="id").mismatched.iloc[0]

    assert (row["id"], row["left_index"], row["right_index"]) == ("b", 11, 20)
    assert (row["x_left"], row["x_right"], row["x_agree"]) == (2.0, 5.0, False)


def test_repeated_keys_are_not_scored():
    """A key repeated on either side is ambiguous, not a pile of disagreements"""
    left = pd.DataFrame({"id": ["a", "a", "b", "c"], "x": [1.0, 2.0, 3.0, 4.0]})
    right = pd.DataFrame({"id": ["a", "a", "b", "c", "c"], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})

    # The naive join scores 4 "a" rows and 2 "c" rows, 3 of them as disagreements
    naive = left.merge(right, on="id", suffixes=("_left", "_right"))
    assert (naive["x_left"] != naive["x_right"]).sum() == 3

    result = compare_keyed(left, right, on="id")
    assert list(result.pairs["id"]) == ["b"]
    assert result.mismatched.empty
    assert result.ambiguous.to_dict("records") == [
        {"id": "a", "n_left": 2, "n_right": 2},
        {"id": "c", "n_left": 1, "n_right": 2},
    ]
    assert result.left_only.empty and result.right_only.empty


def test_repeated_key_on_one_side_only_stays_one_sided():
    """A repeated key the other side lacks is one-sided rows, not ambiguous"""
    left = pd.DataFrame({"id": ["a", "a"], "x": [1.0, 2.0]})
    right = pd.DataFrame({"id": ["b"], "x": [1.0]})
    result = compare_keyed(left, right, on="id")

    assert len(result.left_only) == 2
    assert result.ambiguous.empty


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


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
