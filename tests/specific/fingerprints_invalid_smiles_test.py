"""Tests for resilient Morgan fingerprint generation."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("rdkit") is None,
    reason="pandas and rdkit are required for fingerprint tests",
)


def _test_deps():
    import pandas as pd

    from workbench.utils.chem_utils.fingerprints import compute_morgan_fingerprints

    return pd, compute_morgan_fingerprints


def test_invalid_smiles_rows_are_preserved():
    pd, compute_morgan_fingerprints = _test_deps()
    df = pd.DataFrame(
        {
            "id": ["bad", "good", "blank"],
            "smiles": ["INVALID", "CCO", ""],
        }
    )

    out = compute_morgan_fingerprints(df)

    assert out["id"].tolist() == ["bad", "good", "blank"]
    assert pd.isna(out.loc[out["id"] == "bad", "fingerprint"]).iloc[0]
    assert isinstance(out.loc[out["id"] == "good", "fingerprint"].iloc[0], str)
    assert pd.isna(out.loc[out["id"] == "blank", "fingerprint"]).iloc[0]
    assert "molecule" not in out.columns


def test_existing_invalid_molecule_rows_are_preserved():
    pd, compute_morgan_fingerprints = _test_deps()
    from rdkit import Chem

    df = pd.DataFrame(
        {
            "id": ["good", "bad"],
            "SMILES": ["CCO", "INVALID"],
            "molecule": [Chem.MolFromSmiles("CCO"), None],
        }
    )

    out = compute_morgan_fingerprints(df)

    assert out["id"].tolist() == ["good", "bad"]
    assert isinstance(out.loc[out["id"] == "good", "fingerprint"].iloc[0], str)
    assert pd.isna(out.loc[out["id"] == "bad", "fingerprint"]).iloc[0]
    assert "molecule" in out.columns
