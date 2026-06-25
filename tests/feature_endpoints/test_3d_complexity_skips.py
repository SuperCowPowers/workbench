"""Complexity-guard regression tests for the 3D descriptor endpoint.

Pulls a curated set of SMILES designed to exercise each specific skip path
(skip:heavy_atoms, skip:rot_bonds, skip:rings, skip:ring_complexity,
skip:embed, skip:cost) plus 'ok' controls, then hits the endpoint and asserts
every compound's desc3d_status matches the expected value.

Changes to complexity thresholds in mol_descriptors_3d.py would shift these
expected values — update the reference set rather than the thresholds alone.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/complexity_skips

Populated by: scripts/admin/populate_complexity_skip_reference_compounds.py
"""

import pytest

from workbench.api import Endpoint, PublicData

ENDPOINT_NAME = "smiles-to-3d-fast-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/complexity_skips"


@pytest.fixture(scope="module")
def endpoint():
    return Endpoint(ENDPOINT_NAME)


@pytest.fixture(scope="module")
def reference_df():
    ref_df = PublicData().get(REFERENCE_DATASET)
    assert ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return ref_df


@pytest.fixture(scope="module")
def result_df(endpoint, reference_df):
    payload = reference_df[["id", "smiles"]].copy()
    pred = endpoint.inference(payload)
    return pred.merge(
        reference_df[["id", "name", "expected_status", "notes"]],
        on="id",
        how="left",
    )


def test_endpoint_exists(endpoint):
    assert endpoint.exists(), f"Endpoint '{ENDPOINT_NAME}' does not exist"


def test_reference_compounds_loadable(reference_df):
    assert len(reference_df) > 0
    assert {"id", "name", "smiles", "expected_status"}.issubset(reference_df.columns)


def test_desc3d_status_matches_expected(result_df):
    """Every reference compound's desc3d_status matches expected_status."""
    failures = []
    for _, row in result_df.iterrows():
        actual = row["desc3d_status"]
        expected = row["expected_status"]
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: desc3d_status={actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} compounds have wrong desc3d_status"


if __name__ == "__main__":
    # Build the fixture objects directly for standalone runs.
    ep = Endpoint(ENDPOINT_NAME)
    ref_df = PublicData().get(REFERENCE_DATASET)
    assert ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    res_df = ep.inference(ref_df[["id", "smiles"]].copy()).merge(
        ref_df[["id", "name", "expected_status", "notes"]],
        on="id",
        how="left",
    )

    test_endpoint_exists(ep)
    test_reference_compounds_loadable(ref_df)
    test_desc3d_status_matches_expected(res_df)
    print("\nAll 3D complexity-skip tests passed!")
