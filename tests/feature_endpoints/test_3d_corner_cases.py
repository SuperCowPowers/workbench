"""Corner-case input handling tests for the 3D descriptor endpoint.

Pulls a curated set of edge-case SMILES inputs (empty string, invalid parse,
overly long, isotopes, unusual atoms) and asserts the endpoint handles each
gracefully — producing the expected desc3d_status without crashing.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/corner_cases

Populated by: scripts/admin/populate_reference_compounds.py --dataset corner_cases
"""

import pytest

from workbench.api import Endpoint, PublicData

# Hits the async full endpoint (real conformer ensembles + cold-start warm-up):
# long tier, excluded from quick runs.
pytestmark = pytest.mark.long

ENDPOINT_NAME = "smiles-to-3d-full-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/corner_cases"


# ---------------------------------------------------------------------------
# Fixtures: module-scoped so AWS calls happen once per file, at run time
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def endpoint():
    return Endpoint(ENDPOINT_NAME)


@pytest.fixture(scope="module")
def reference_df():
    df = PublicData().get(REFERENCE_DATASET)
    assert df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return df


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


def test_endpoint_does_not_crash(result_df, reference_df):
    """Endpoint returns a row per input, even for malformed SMILES."""
    assert len(result_df) == len(reference_df), "Endpoint dropped rows — it should return NaN features instead"


def test_desc3d_status_matches_expected(result_df):
    """Every corner-case input produces the expected desc3d_status."""
    failures = []
    for _, row in result_df.iterrows():
        actual = row["desc3d_status"]
        expected = row["expected_status"]
        if actual != expected:
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: desc3d_status={actual!r}, expected {expected!r}")
    assert not failures, f"{len(failures)} corner cases have wrong desc3d_status"


if __name__ == "__main__":
    # Construct artifacts and data directly, then pass to tests explicitly
    ep = Endpoint(ENDPOINT_NAME)
    ref = PublicData().get(REFERENCE_DATASET)
    pred = ep.inference(ref[["id", "smiles"]].copy())
    result = pred.merge(ref[["id", "name", "expected_status", "notes"]], on="id", how="left")

    test_endpoint_exists(ep)
    test_reference_compounds_loadable(ref)
    test_endpoint_does_not_crash(result, ref)
    test_desc3d_status_matches_expected(result)
    print("\nAll 3D corner-case tests passed!")
