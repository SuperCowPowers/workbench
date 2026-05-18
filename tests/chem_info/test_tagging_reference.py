"""Tagging regression tests.

Pulls a public reference set of input SMILES paired with the expected
curation tags emitted by:

    workbench.utils.chem_utils.mol_standardize.standardize  ->
    workbench.utils.chem_utils.mol_tagging.tag_molecules

then runs the pipeline and asserts every expected curation tag fires (and
nothing extra). Complements the standardization reference test — that one
gates structure-handling changes; this one gates ADMET-curation policy
changes.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/tagging

Populated by: data/public_data/build_tagging_reference.py
"""

import pandas as pd

from workbench.api import PublicData
from workbench.utils.chem_utils.mol_standardize import standardize
from workbench.utils.chem_utils.mol_tagging import tag_molecules

REFERENCE_DATASET = "comp_chem/reference_compounds/tagging"

_ref_df = None
_result_df = None


def _parse_expected_tags(value) -> list[str]:
    """expected_curation_tags is stored as pipe-delimited; NaN/empty -> []."""
    if pd.isna(value) or value == "":
        return []
    return sorted(value.split("|"))


def _get_reference_df() -> pd.DataFrame:
    global _ref_df
    if _ref_df is None:
        _ref_df = PublicData().get(REFERENCE_DATASET)
        assert _ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return _ref_df


def _get_result() -> pd.DataFrame:
    """Run standardize() + tag_molecules() on the reference inputs (cached)
    and join back to expected-value columns."""
    global _result_df
    if _result_df is None:
        ref_df = _get_reference_df()
        input_df = ref_df.rename(columns={"input_smiles": "smiles"})[["id", "name", "smiles"]].copy()
        std_df = standardize(input_df)
        tagged = tag_molecules(std_df)
        _result_df = tagged.merge(
            ref_df[
                [
                    "id",
                    "standardized_smiles",
                    "expected_undefined_chiral_centers",
                    "expected_curation_tags",
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
        "standardized_smiles",
        "expected_undefined_chiral_centers",
        "expected_curation_tags",
        "notes",
    }
    assert expected_cols.issubset(ref_df.columns)
    print(f"Loaded {len(ref_df)} reference compounds")


def test_standardized_smiles_match():
    """Standardized SMILES output matches the reference snapshot."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = row["standardized_smiles"]
        actual = row["smiles"]
        if pd.isna(expected) or expected == "":
            continue
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: got {actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} standardized SMILES mismatches"


def test_undefined_chiral_centers_match():
    """undefined_chiral_centers from standardize() matches the snapshot."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = row["expected_undefined_chiral_centers"]
        if pd.isna(expected):
            continue
        actual = row.get("undefined_chiral_centers")
        if actual is None or pd.isna(actual):
            failures.append((row["name"], None, int(expected)))
            continue
        if int(actual) != int(expected):
            failures.append((row["name"], int(actual), int(expected)))

    for name, actual, expected in failures:
        print(f"  {name}: got {actual}, expected {expected}")
    assert not failures, f"{len(failures)} undefined_chiral_centers mismatches"


def test_curation_tags_match():
    """The curation:* tags emitted by tag_molecules() match expected_curation_tags exactly."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = _parse_expected_tags(row["expected_curation_tags"])
        actual = sorted(t for t in row["tags"] if t.startswith("curation:"))
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}:")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            print(f"    missing:  {missing}")
        if extra:
            print(f"    extra:    {extra}")
    assert not failures, f"{len(failures)} curation-tag mismatches"


if __name__ == "__main__":
    test_reference_compounds_loadable()
    test_standardized_smiles_match()
    test_undefined_chiral_centers_match()
    test_curation_tags_match()
    print("\nAll tagging reference tests passed.")
