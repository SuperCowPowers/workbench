"""Complexity-guard regression tests for the 3D descriptor endpoint.

Pulls a curated set of SMILES designed to exercise each specific skip path
(skip:heavy_atoms, skip:rot_bonds, skip:rings, skip:ring_complexity,
skip:embed) plus 'ok' controls, then hits the endpoint and asserts every
compound's desc3d_status matches the expected value.

Changes to complexity thresholds in mol_descriptors_3d.py would shift these
expected values — update the reference set rather than the thresholds alone.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/complexity_skips

Populated by: scripts/admin/populate_complexity_skip_reference_compounds.py
"""

import pandas as pd

from workbench.api import Endpoint, PublicData

ENDPOINT_NAME = "smiles-to-3d-fast-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/complexity_skips"

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


def test_desc3d_status_matches_expected():
    """Every reference compound's desc3d_status matches expected_status."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        actual = row["desc3d_status"]
        expected = row["expected_status"]
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: desc3d_status={actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} compounds have wrong desc3d_status"


if __name__ == "__main__":
    test_endpoint_exists()
    test_reference_compounds_loadable()
    test_desc3d_status_matches_expected()
    print("\nAll 3D complexity-skip tests passed!")
