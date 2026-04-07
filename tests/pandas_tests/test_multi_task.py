"""Tests for workbench.utils.multi_task."""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.multi_task import combine_multi_task_data, validate_multi_task_data


def _make_df(ids, smiles, features, targets):
    """Helper to build a DataFrame with id, smiles, feature columns, and target columns.

    Args:
        ids: list of id values
        smiles: list of SMILES strings
        features: dict of {col_name: values}
        targets: dict of {col_name: values}
    """
    data = {"id": ids, "smiles": smiles, **features, **targets}
    return pd.DataFrame(data)


# --- Core functionality ---


def test_basic_no_overlap():
    """Two DataFrames with no shared molecules concat all rows."""
    df1 = _make_df(
        ids=["A", "B"],
        smiles=["CC", "CCC"],
        features={"feat1": [1.0, 2.0], "feat2": [10.0, 20.0]},
        targets={"ppb": [0.5, 0.6]},
    )
    df2 = _make_df(
        ids=["C", "D"],
        smiles=["CCCC", "CCCCC"],
        features={"feat1": [3.0, 4.0], "feat2": [30.0, 40.0]},
        targets={"logd": [1.1, 2.2]},
    )

    result = combine_multi_task_data([df1, df2], [["ppb"], ["logd"]])

    assert len(result) == 4
    assert set(result.columns) == {"id", "smiles", "feat1", "feat2", "ppb", "logd"}
    # Molecules from df1 should have NaN logd
    assert result.loc[result["id"] == "A", "logd"].isna().all()
    # Molecules from df2 should have NaN ppb
    assert result.loc[result["id"] == "C", "ppb"].isna().all()


def test_with_overlap_collapses_rows():
    """Shared molecules are collapsed so both targets appear on one row."""
    df1 = _make_df(
        ids=["A", "B"],
        smiles=["CC", "CCC"],
        features={"feat1": [1.0, 2.0]},
        targets={"ppb": [0.5, 0.6]},
    )
    df2 = _make_df(
        ids=["B", "C"],
        smiles=["CCC", "CCCC"],
        features={"feat1": [2.0, 3.0]},
        targets={"logd": [1.1, 2.2]},
    )

    result = combine_multi_task_data([df1, df2], [["ppb"], ["logd"]])

    assert len(result) == 3  # A, B, C
    row_b = result[result["id"] == "B"].iloc[0]
    assert row_b["ppb"] == 0.6
    assert row_b["logd"] == 1.1


def test_multiple_targets_per_df():
    """A DataFrame can contribute multiple target columns."""
    df1 = _make_df(
        ids=["A", "B"],
        smiles=["CC", "CCC"],
        features={"feat1": [1.0, 2.0]},
        targets={"ppb_human": [0.5, 0.6], "ppb_mouse": [0.7, 0.8]},
    )
    df2 = _make_df(
        ids=["C"],
        smiles=["CCCC"],
        features={"feat1": [3.0]},
        targets={"logd": [1.1]},
    )

    result = combine_multi_task_data([df1, df2], [["ppb_human", "ppb_mouse"], ["logd"]])

    assert len(result) == 3
    assert "ppb_human" in result.columns
    assert "ppb_mouse" in result.columns
    assert "logd" in result.columns


def test_shared_features_intersection():
    """Only columns present in ALL DataFrames are kept as features."""
    df1 = _make_df(
        ids=["A"],
        smiles=["CC"],
        features={"shared_feat": [1.0], "only_in_df1": [99.0]},
        targets={"t1": [0.5]},
    )
    df2 = _make_df(
        ids=["B"],
        smiles=["CCC"],
        features={"shared_feat": [2.0], "only_in_df2": [88.0]},
        targets={"t2": [0.6]},
    )

    result = combine_multi_task_data([df1, df2], [["t1"], ["t2"]])

    assert "shared_feat" in result.columns
    assert "only_in_df1" not in result.columns
    assert "only_in_df2" not in result.columns


def test_custom_id_column():
    """Custom id_column is respected."""
    df1 = pd.DataFrame({"mol_id": ["A"], "smiles": ["CC"], "f": [1.0], "t1": [0.5]})
    df2 = pd.DataFrame({"mol_id": ["B"], "smiles": ["CCC"], "f": [2.0], "t2": [0.6]})

    result = combine_multi_task_data([df1, df2], [["t1"], ["t2"]], id_column="mol_id")

    assert len(result) == 2
    assert "mol_id" in result.columns


def test_three_dataframes():
    """Rows from all DataFrames are concatenated."""
    common = {"feat1": [1.0], "feat2": [2.0]}
    df1 = _make_df(ids=["A"], smiles=["CC"], features=common, targets={"t1": [0.1]})
    df2 = _make_df(ids=["B"], smiles=["CCC"], features=common, targets={"t2": [0.2]})
    df3 = _make_df(ids=["C"], smiles=["CCCC"], features=common, targets={"t3": [0.3]})

    result = combine_multi_task_data([df1, df2, df3], [["t1"], ["t2"], ["t3"]])

    assert len(result) == 3
    assert {"t1", "t2", "t3"}.issubset(result.columns)


def test_nan_targets_from_missing_columns():
    """Targets not in a source DataFrame become NaN after concat."""
    df1 = _make_df(
        ids=["A"],
        smiles=["CC"],
        features={"feat1": [1.0]},
        targets={"t1": [0.5]},
    )
    df2 = _make_df(
        ids=["B"],
        smiles=["CCC"],
        features={"feat1": [2.0]},
        targets={"t2": [0.6]},
    )

    result = combine_multi_task_data([df1, df2], [["t1"], ["t2"]])

    row_a = result[result["id"] == "A"].iloc[0]
    assert row_a["t1"] == 0.5
    assert np.isnan(row_a["t2"])
    row_b = result[result["id"] == "B"].iloc[0]
    assert np.isnan(row_b["t1"])
    assert row_b["t2"] == 0.6


def test_duplicate_ids_averaged():
    """Duplicate IDs within a source are collapsed by averaging numeric values."""
    df1 = _make_df(
        ids=["A", "A"],
        smiles=["CC", "CC"],
        features={"feat1": [1.0, 1.0]},
        targets={"t1": [0.5, 0.9]},
    )
    df2 = _make_df(
        ids=["B"],
        smiles=["CCC"],
        features={"feat1": [2.0]},
        targets={"t2": [0.6]},
    )

    result = combine_multi_task_data([df1, df2], [["t1"], ["t2"]])

    assert len(result) == 2  # A (collapsed) + B
    assert result[result["id"] == "A"].iloc[0]["t1"] == pytest.approx(0.7)  # Mean of 0.5 and 0.9


# --- Input validation ---


def test_error_length_mismatch():
    df = pd.DataFrame({"id": ["A"], "smiles": ["CC"], "t": [1.0]})
    with pytest.raises(ValueError, match="same length"):
        combine_multi_task_data([df], [["t"], ["extra"]])


def test_error_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        combine_multi_task_data([], [])


def test_error_missing_id_column():
    df = pd.DataFrame({"smiles": ["CC"], "t": [1.0]})
    with pytest.raises(ValueError, match="missing id_column"):
        combine_multi_task_data([df], [["t"]])


def test_error_missing_smiles():
    df = pd.DataFrame({"id": ["A"], "t": [1.0]})
    with pytest.raises(ValueError, match="missing 'smiles'"):
        combine_multi_task_data([df], [["t"]])


def test_error_missing_target_column():
    df = pd.DataFrame({"id": ["A"], "smiles": ["CC"], "f": [1.0]})
    with pytest.raises(ValueError, match="missing target columns"):
        combine_multi_task_data([df], [["nonexistent"]])


def test_error_duplicate_target_names():
    df1 = pd.DataFrame({"id": ["A"], "smiles": ["CC"], "t": [1.0]})
    df2 = pd.DataFrame({"id": ["B"], "smiles": ["CCC"], "t": [2.0]})
    with pytest.raises(ValueError, match="Duplicate target"):
        combine_multi_task_data([df1, df2], [["t"], ["t"]])


# --- validate_multi_task_data ---


def test_validate_passes_clean_data():
    """Clean data passes validation without error."""
    df = _make_df(
        ids=["A", "B"],
        smiles=["CC", "CCC"],
        features={"feat1": [1.0, 2.0]},
        targets={"t1": [0.5, 0.6]},
    )

    validate_multi_task_data(df, ["t1"])  # Should not raise


def test_validate_catches_null_ids():
    df = pd.DataFrame(
        {
            "id": ["A", None],
            "smiles": ["CC", "CCC"],
            "feat1": [1.0, 2.0],
            "t1": [0.5, 0.6],
        }
    )

    with pytest.raises(ValueError, match="NaN values"):
        validate_multi_task_data(df, ["t1"])


def test_validate_catches_duplicate_ids():
    df = _make_df(
        ids=["A", "A"],
        smiles=["CC", "CCC"],
        features={"feat1": [1.0, 2.0]},
        targets={"t1": [0.5, 0.6]},
    )

    with pytest.raises(ValueError, match="duplicate values"):
        validate_multi_task_data(df, ["t1"])


def test_validate_catches_missing_smiles():
    df = pd.DataFrame(
        {
            "id": ["A"],
            "feat1": [1.0],
            "t1": [0.5],
        }
    )

    with pytest.raises(ValueError, match="smiles"):
        validate_multi_task_data(df, ["t1"])


def test_validate_catches_missing_target():
    df = _make_df(
        ids=["A"],
        smiles=["CC"],
        features={"feat1": [1.0]},
        targets={"t1": [0.5]},
    )

    with pytest.raises(ValueError, match="missing"):
        validate_multi_task_data(df, ["t1", "t_nonexistent"])


def test_validate_catches_empty_target():
    df = _make_df(
        ids=["A"],
        smiles=["CC"],
        features={"feat1": [1.0]},
        targets={"t1": [np.nan]},
    )

    with pytest.raises(ValueError, match="zero non-null"):
        validate_multi_task_data(df, ["t1"])


# --- Two-pass merge (id-based then smiles-based) ---


def test_two_pass_merge():
    """Simulates the pipeline pattern: merge on ID first, then on SMILES for external data."""
    # Pass 1: Two non-overlapping sources
    df_ppb = _make_df(
        ids=["A", "B"],
        smiles=["CC", "CCC"],
        features={"feat1": [1.0, 2.0]},
        targets={"ppb": [0.5, 0.6]},
    )
    df_logd = _make_df(
        ids=["C", "D"],
        smiles=["CCCC", "CCCCC"],
        features={"feat1": [3.0, 4.0]},
        targets={"logd": [1.1, 2.2]},
    )
    merged = combine_multi_task_data([df_ppb, df_logd], [["ppb"], ["logd"]])
    assert len(merged) == 4

    # Pass 2: External source, merge on SMILES
    df_logp = pd.DataFrame(
        {
            "id": ["X1", "X2"],
            "smiles": ["CC", "CCCCCC"],  # CC=A, CCCCCC=new
            "feat1": [1.0, 5.0],
            "logp": [10.0, 50.0],
        }
    )
    result = combine_multi_task_data([merged, df_logp], [["ppb", "logd"], ["logp"]], merge_on_smiles=True)

    assert len(result) == 5  # CC, CCC, CCCC, CCCCC, CCCCCC
    # CC (molecule A) gets ppb + logp collapsed onto one row
    row_cc = result[result["smiles"] == "CC"].iloc[0]
    assert row_cc["ppb"] == 0.5
    assert row_cc["logp"] == 10.0
    # CCCCCC is logp-only
    row_new = result[result["smiles"] == "CCCCCC"].iloc[0]
    assert np.isnan(row_new["ppb"])
    assert np.isnan(row_new["logd"])
    assert row_new["logp"] == 50.0
