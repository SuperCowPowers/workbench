"""Tests for workbench.utils.multi_task.

These are hermetic/offline unit tests: the pure helpers (combine, validate,
weights) run on synthetic DataFrames, and the FeatureSet-pulling path is
exercised by injecting a fake ``workbench.api`` module into ``sys.modules`` so no
AWS connectivity is required. The focus is the multi-task merge mechanics and the
temporal-date handling — in particular the id-strict / smiles-lenient rule for a
source that lacks the date column.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from workbench.utils.multi_task import (
    combine_multi_task_data,
    compute_inverse_count_task_weights,
    pull_multi_task_data,
    validate_multi_task_data,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _install_fake_feature_set(monkeypatch, frames: dict):
    """Inject a fake ``workbench.api`` exposing FeatureSet backed by ``frames``.

    pull_multi_task_data does a lazy ``from workbench.api import FeatureSet``, so
    seeding sys.modules with a stand-in avoids importing the real (AWS-bound)
    workbench.api entirely. ``frames`` maps FeatureSet name -> DataFrame.
    """
    fake_api = types.ModuleType("workbench.api")

    class _FakeFeatureSet:
        def __init__(self, name):
            if name not in frames:
                raise KeyError(f"unknown FeatureSet '{name}'")
            self._df = frames[name]

        def pull_dataframe(self):
            return self._df.copy()

    fake_api.FeatureSet = _FakeFeatureSet
    monkeypatch.setitem(sys.modules, "workbench.api", fake_api)


# ----------------------------------------------------------------------------
# compute_inverse_count_task_weights
# ----------------------------------------------------------------------------
def test_inverse_count_weights_basic():
    """Weights are inversely proportional to non-NaN counts and mean-normalize to 1."""
    targets = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, 5.0, np.nan],
            [np.nan, 6.0, np.nan],
            [3.0, 7.0, 9.0],
            [np.nan, np.nan, 10.0],
        ]
    )  # per-task non-NaN counts: [3, 3, 2]
    w = compute_inverse_count_task_weights(targets)

    assert w.dtype == np.float32
    assert w.shape == (3,)
    assert w.mean() == pytest.approx(1.0, rel=1e-6)
    # Equal-count tasks get equal weight; the rarer task is up-weighted.
    assert w[0] == pytest.approx(w[1])
    assert w[2] > w[0]


def test_inverse_count_weights_zero_count_raises():
    """A task with zero non-NaN rows is an error (cannot normalize)."""
    targets = np.array([[1.0, np.nan], [2.0, np.nan]])  # second task all-NaN
    with pytest.raises(ValueError, match="at least one non-NaN"):
        compute_inverse_count_task_weights(targets)


def test_inverse_count_weights_requires_2d():
    with pytest.raises(ValueError, match="must be 2D"):
        compute_inverse_count_task_weights(np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------------
# combine_multi_task_data
# ----------------------------------------------------------------------------
def test_combine_id_merge_partial_overlap():
    """ID-based merge: shared molecules collapse to one row; missing targets are NaN."""
    df_p = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "smiles": ["CC", "CCC", "CCCC", "CCCCC"],
            "primary": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df_a = pd.DataFrame(
        {
            "id": [3, 4, 5, 6],
            "smiles": ["CCCC", "CCCCC", "C1CC1", "C1CCC1"],
            "aux": [33.0, 44.0, 55.0, 66.0],
        }
    )
    out = combine_multi_task_data([df_p, df_a], [["primary"], ["aux"]], id_column="id")

    assert set(out.columns) == {"id", "smiles", "primary", "aux"}
    assert len(out) == 6  # union of ids 1..6
    row3 = out[out["id"] == 3].iloc[0]
    assert row3["primary"] == 30.0 and row3["aux"] == 33.0  # both targets present
    row1 = out[out["id"] == 1].iloc[0]
    assert row1["primary"] == 10.0 and pd.isna(row1["aux"])  # aux missing -> NaN


def test_combine_empty_target_raises():
    """A target column that ends up all-NaN is a hard error."""
    df_p = pd.DataFrame({"id": [1, 2], "smiles": ["CC", "CCC"], "primary": [1.0, 2.0]})
    df_a = pd.DataFrame({"id": [1, 2], "smiles": ["CC", "CCC"], "aux": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="ALL NaN"):
        combine_multi_task_data([df_p, df_a], [["primary"], ["aux"]], id_column="id")


def test_combine_passthrough_not_treated_as_target():
    """Passthrough columns survive the merge but are excluded from target validation.

    An all-NaN *target* raises, but an all-NaN *passthrough* must not — this is the
    mechanism that lets a date-less source carry a NaT date column through. Mirrors
    real usage where each source's date is a private, uniquely-named column.
    """
    df_p = pd.DataFrame(
        {"id": [1, 2], "smiles": ["CC", "CCC"], "primary": [1.0, 2.0], "__date_p": ["2024-01-01", "2024-02-01"]}
    )
    df_a = pd.DataFrame(
        {
            "id": [2, 3],
            "smiles": ["CCC", "CCCC"],
            "aux": [3.0, 4.0],
            "__date_a": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),  # date-less source
        }
    )
    # Should not raise even though __date_a is entirely NaT (passthrough, not target).
    out = combine_multi_task_data(
        [df_p, df_a],
        [["primary"], ["aux"]],
        id_column="id",
        passthrough_columns=[["__date_p"], ["__date_a"]],
    )
    assert {"__date_p", "__date_a"}.issubset(out.columns)
    assert out["__date_a"].isna().all()  # all-NaN passthrough survived without error
    assert pd.to_datetime(out.loc[out["id"] == 1, "__date_p"].iloc[0]) == pd.Timestamp("2024-01-01")


def test_combine_smiles_merge():
    """SMILES-based merge joins external data with no shared id namespace."""
    internal = pd.DataFrame({"id": [1, 2], "smiles": ["CCO", "CCC"], "primary": [10.0, 20.0]})
    external = pd.DataFrame({"id": [99, 98], "smiles": ["CCO", "CCCC"], "aux": [1.5, 2.5]})
    # standardize_smiles=False keeps this test pure-pandas (no rdkit dependency).
    out = combine_multi_task_data(
        [internal, external],
        [["primary"], ["aux"]],
        id_column="id",
        merge_on_smiles=True,
        standardize_smiles=False,
    )
    cco = out[out["smiles"] == "CCO"].iloc[0]
    assert cco["primary"] == 10.0 and cco["aux"] == 1.5  # joined on SMILES
    ext_only = out[out["smiles"] == "CCCC"].iloc[0]
    assert pd.isna(ext_only["primary"]) and ext_only["aux"] == 2.5


# ----------------------------------------------------------------------------
# validate_multi_task_data
# ----------------------------------------------------------------------------
def test_validate_clean_passes():
    df = pd.DataFrame({"udm_mol_bat_id": ["1", "2", "3"], "smiles": ["CC", "CCC", "CCCC"], "t": [1.0, np.nan, 3.0]})
    # Should not raise
    validate_multi_task_data(df, ["t"], id_column="udm_mol_bat_id")


def test_validate_duplicate_ids_raise():
    df = pd.DataFrame({"udm_mol_bat_id": ["1", "1"], "smiles": ["CC", "CCC"], "t": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_multi_task_data(df, ["t"], id_column="udm_mol_bat_id")


def test_validate_missing_target_raises():
    df = pd.DataFrame({"udm_mol_bat_id": ["1", "2"], "smiles": ["CC", "CCC"], "t": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing"):
        validate_multi_task_data(df, ["t", "does_not_exist"], id_column="udm_mol_bat_id")


# ----------------------------------------------------------------------------
# pull_multi_task_data — date handling (the heart of the recent change)
# ----------------------------------------------------------------------------
def test_pull_date_synthesis_row_wise_max(monkeypatch):
    """Canonical date_col is the row-wise max of per-source dates (leakage-safe)."""
    frames = {
        "src_a": pd.DataFrame(
            {
                "udm_mol_bat_id": ["1", "2"],
                "smiles": ["CC", "CCC"],
                "ta": [10.0, 20.0],
                "udm_asy_date": ["2024-01-01", "2025-12-01"],
            }
        ),
        "src_b": pd.DataFrame(
            {
                "udm_mol_bat_id": ["2", "3"],
                "smiles": ["CCC", "CCCC"],
                "tb": [30.0, 40.0],
                "udm_asy_date": ["2025-12-31", "2023-01-01"],
            }
        ),
    }
    _install_fake_feature_set(monkeypatch, frames)

    out = pull_multi_task_data(
        {"src_a": {"target_info": ["ta"]}, "src_b": {"target_info": ["tb"]}},
        id_column="udm_mol_bat_id",
        date_col="udm_asy_date",
    )

    # Synthesized date is a plain YYYY-MM-DD string (not datetime64) so it stores
    # verbatim via PandasToFeatures and round-trips through a tz-naive temporal split.
    assert out["udm_asy_date"].dtype == object
    dates = dict(zip(out["udm_mol_bat_id"], out["udm_asy_date"]))
    # id 2 is in both sources -> max(2025-12-01, 2025-12-31) = 2025-12-31
    assert dates["2"] == "2025-12-31"
    assert dates["1"] == "2024-01-01"
    assert dates["3"] == "2023-01-01"
    # No leftover private date columns
    assert not [c for c in out.columns if c.startswith("__date_")]


def test_pull_id_source_missing_date_raises(monkeypatch):
    """An ID-based (internal) source missing date_col fails loudly — likely a typo."""
    frames = {
        "src_a": pd.DataFrame({"udm_mol_bat_id": ["1"], "smiles": ["CC"], "ta": [1.0], "udm_asy_date": ["2024-01-01"]}),
        "src_no_date": pd.DataFrame({"udm_mol_bat_id": ["1"], "smiles": ["CC"], "tb": [2.0]}),
    }
    _install_fake_feature_set(monkeypatch, frames)

    with pytest.raises(ValueError, match="missing required date_col"):
        pull_multi_task_data(
            {"src_a": {"target_info": ["ta"]}, "src_no_date": {"target_info": ["tb"]}},
            id_column="udm_mol_bat_id",
            date_col="udm_asy_date",
        )


def test_pull_smiles_source_missing_date_is_training(monkeypatch):
    """A SMILES-based (public) source missing date_col contributes NaT (training-only)."""
    pytest.importorskip("rdkit")  # the smiles pass standardizes SMILES
    from workbench.utils.pandas_utils import temporal_split

    frames = {
        "ppb": pd.DataFrame(
            {
                "udm_mol_bat_id": ["1", "2"],
                "smiles": ["CCO", "CCC"],
                "ppb_human": [10.0, 20.0],
                "udm_asy_date": ["2024-01-01", "2026-01-01"],  # id 2 is post-cutoff
            }
        ),
        # public LogP: 'id' identifier, no udm_asy_date
        "logp_public": pd.DataFrame(
            {"id": ["a", "b", "c"], "smiles": ["CCO", "CCCC", "CCCCC"], "logp": [1.0, 2.0, 3.0]}
        ),
    }
    _install_fake_feature_set(monkeypatch, frames)

    out = pull_multi_task_data(
        {"ppb": {"target_info": ["ppb_human"]}},
        {"logp_public": {"target_info": ["logp"], "src_id_col": "id"}},
        id_column="udm_mol_bat_id",
        date_col="udm_asy_date",
    )

    # Public-only molecules (CCCC, CCCCC) have no date -> NaT
    public_only = out[out["logp"].notna() & out["ppb_human"].isna()]
    assert len(public_only) == 2
    assert public_only["udm_asy_date"].isna().all()

    # The CCO molecule overlaps the internal compound -> carries the internal date
    cco = out[out["smiles"].str.contains("O", na=False)].iloc[0]
    assert cco["ppb_human"] == 10.0 and cco["logp"] == 1.0
    assert pd.to_datetime(cco["udm_asy_date"]) == pd.Timestamp("2024-01-01")

    # Temporal split: only the post-cutoff internal compound is held out; the
    # date-less public rows fall to the training side (NaT is neither <= nor > cutoff).
    _train, holdout = temporal_split(out, "udm_asy_date", end_date="2025-10-17")
    assert set(holdout["udm_mol_bat_id"]) == {"2"}
