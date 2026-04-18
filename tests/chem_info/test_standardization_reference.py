"""Standardization regression tests.

Pulls a public reference set of input SMILES paired with the expected outputs
of workbench.utils.chem_utils.mol_standardize.standardize(), then runs the
pipeline and asserts every expected value matches.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/standardization

Populated by: scripts/admin/populate_standardization_reference_compounds.py
"""

import pandas as pd

from workbench.api import PublicData
from workbench.utils.chem_utils.mol_standardize import standardize

REFERENCE_DATASET = "comp_chem/reference_compounds/standardization"

_ref_df = None
_result_df = None


def _get_reference_df() -> pd.DataFrame:
    global _ref_df
    if _ref_df is None:
        _ref_df = PublicData().get(REFERENCE_DATASET)
        assert _ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return _ref_df


def _get_result() -> pd.DataFrame:
    """Run standardize() on the reference inputs (cached) and join back to expected values."""
    global _result_df
    if _result_df is None:
        ref_df = _get_reference_df()
        # standardize expects a column literally named "smiles"
        input_df = ref_df.rename(columns={"input_smiles": "smiles"})[["id", "name", "smiles"]].copy()
        out = standardize(input_df)
        # Join expected-value columns alongside the actual output
        _result_df = out.merge(
            ref_df[
                [
                    "id",
                    "expected_smiles",
                    "expected_salt",
                    "expected_undefined_chiral_centers",
                    "notes",
                ]
            ],
            on="id",
            how="left",
        )
    return _result_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reference_compounds_loadable():
    ref_df = _get_reference_df()
    assert len(ref_df) > 0
    expected_cols = {
        "id",
        "name",
        "input_smiles",
        "expected_smiles",
        "expected_salt",
        "expected_undefined_chiral_centers",
    }
    assert expected_cols.issubset(ref_df.columns)
    print(f"Loaded {len(ref_df)} reference compounds")


def test_standardized_smiles_match():
    """The 'smiles' column from standardize() matches expected_smiles for every row."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = row["expected_smiles"]
        actual = row["smiles"]
        if pd.isna(expected) or expected == "":
            continue
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: got {actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} standardized SMILES mismatches"


def test_extracted_salts_match():
    """The 'salt' column matches expected_salt for every row (None means no salt)."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = row["expected_salt"]
        actual = row["salt"]

        # Normalize None vs NaN vs empty string — any of these means "no salt"
        expected_empty = pd.isna(expected) or expected in (None, "", "None")
        actual_empty = pd.isna(actual) or actual in (None, "", "None")

        if expected_empty and actual_empty:
            continue
        if expected_empty != actual_empty or actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: got salt={actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} salt-extraction mismatches"


def test_undefined_chiral_centers_match():
    """The 'undefined_chiral_centers' column matches expected_undefined_chiral_centers."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = row["expected_undefined_chiral_centers"]
        actual = row["undefined_chiral_centers"]
        if pd.isna(expected):
            continue
        if int(actual) != int(expected):
            failures.append((row["name"], int(actual), int(expected)))

    for name, actual, expected in failures:
        print(f"  {name}: got {actual}, expected {expected}")
    assert not failures, f"{len(failures)} undefined_chiral_centers mismatches"


if __name__ == "__main__":
    test_reference_compounds_loadable()
    test_standardized_smiles_match()
    test_extracted_salts_match()
    test_undefined_chiral_centers_match()
    print("\nAll standardization reference tests passed!")
