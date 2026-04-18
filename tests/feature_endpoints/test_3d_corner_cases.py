"""Corner-case input handling tests for the 3D descriptor endpoint.

Pulls a curated set of edge-case SMILES inputs (empty string, invalid parse,
overly long, isotopes, unusual atoms) and asserts the endpoint handles each
gracefully — producing the expected desc3d_status without crashing.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/corner_cases

Populated by: scripts/admin/populate_corner_cases_reference_compounds.py
"""

import pandas as pd

from workbench.api import Endpoint, PublicData

ENDPOINT_NAME = "smiles-to-3d-descriptors-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/corner_cases"

endpoint = Endpoint(ENDPOINT_NAME)

_ref_df = None
_result_df = None


def _get_reference_df() -> pd.DataFrame:
    global _ref_df
    if _ref_df is None:
        _ref_df = PublicData().get(REFERENCE_DATASET)
        assert _ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return _ref_df


def _get_result() -> pd.DataFrame:
    global _result_df
    if _result_df is None:
        ref_df = _get_reference_df()
        payload = ref_df[["id", "smiles"]].copy()
        pred = endpoint.inference(payload)
        _result_df = pred.merge(
            ref_df[["id", "name", "expected_status", "notes"]],
            on="id",
            how="left",
        )
    return _result_df


def test_endpoint_exists():
    assert endpoint.exists(), f"Endpoint '{ENDPOINT_NAME}' does not exist"


def test_reference_compounds_loadable():
    ref_df = _get_reference_df()
    assert len(ref_df) > 0
    assert {"id", "name", "smiles", "expected_status"}.issubset(ref_df.columns)


def test_endpoint_does_not_crash():
    """Endpoint returns a row per input, even for malformed SMILES."""
    result = _get_result()
    ref_df = _get_reference_df()
    assert len(result) == len(ref_df), "Endpoint dropped rows — it should return NaN features instead"


def test_desc3d_status_matches_expected():
    """Every corner-case input produces the expected desc3d_status."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        actual = row["desc3d_status"]
        expected = row["expected_status"]
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: desc3d_status={actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} corner cases have wrong desc3d_status"


if __name__ == "__main__":
    test_endpoint_exists()
    test_reference_compounds_loadable()
    test_endpoint_does_not_crash()
    test_desc3d_status_matches_expected()
    print("\nAll 3D corner-case tests passed!")
