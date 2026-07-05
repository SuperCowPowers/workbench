"""Fast unit tests for ``workbench.training.validation.split_validation_set``.

Pure pandas — no AWS/config needed, so these run in the default (non-long) suite.
"""

import pandas as pd

# Workbench Imports
from workbench.training.validation import split_validation_set


def _frame(validation_flags):
    return pd.DataFrame({"id": range(len(validation_flags)), "x": 1.0, "validation": validation_flags})


def test_splits_marked_rows():
    """Marked rows go to val_df; the rest to train_df."""
    df = _frame([False, True, False, True])
    train_df, val_df = split_validation_set(df)
    assert sorted(val_df["id"]) == [1, 3]
    assert sorted(train_df["id"]) == [0, 2]


def test_no_marked_rows_is_noop():
    """Nothing marked → all rows train, val_df is empty."""
    df = _frame([False, False, False])
    train_df, val_df = split_validation_set(df)
    assert len(train_df) == 3
    assert len(val_df) == 0


def test_missing_column_is_noop():
    """Absent marker column → train_df is the input, val_df empty (safe to adopt unconditionally)."""
    df = pd.DataFrame({"id": [0, 1], "x": [1.0, 2.0]})
    train_df, val_df = split_validation_set(df)
    assert len(train_df) == 2
    assert len(val_df) == 0


def test_nan_marker_treated_as_false():
    """NaN in the marker column is treated as not-validation."""
    df = _frame([True, None, False])
    train_df, val_df = split_validation_set(df)
    assert sorted(val_df["id"]) == [0]
    assert sorted(train_df["id"]) == [1, 2]


def test_returned_frames_have_reset_index():
    """Both frames come back with a clean 0..n-1 index."""
    df = _frame([False, True, False, True])
    train_df, val_df = split_validation_set(df)
    assert list(train_df.index) == [0, 1]
    assert list(val_df.index) == [0, 1]


def test_custom_marker_column():
    """A non-default marker name is honored."""
    df = pd.DataFrame({"id": [0, 1, 2], "holdout": [False, True, True]})
    train_df, val_df = split_validation_set(df, marker="holdout")
    assert sorted(val_df["id"]) == [1, 2]
    assert sorted(train_df["id"]) == [0]
