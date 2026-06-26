"""Validation tests for the async Boltzmann 3D descriptor endpoint (smiles-to-3d-full-v1).

Pulls a curated set of public reference compounds and runs them through the
async (Boltzmann-ensemble) endpoint, validating geometric invariants and
compound-specific expected ranges. The endpoint uses adaptive conformer
counts (50/300/500 by rotatable-bond tier).

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/reference_compounds_3d

Populated by: scripts/admin/populate_reference_compounds.py --dataset 3d
"""

import pandas as pd
import pytest

from workbench.api import Endpoint, PublicData
from workbench.utils.chem_utils.mol_descriptors_3d import get_3d_feature_names

# Heavy validation study (~6 min): medium tier, excluded from quick runs
pytestmark = pytest.mark.medium

ENDPOINT_NAME = "smiles-to-3d-full-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/reference_compounds_3d"


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: Boltzmann inference is expensive, runs once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def endpoint():
    return Endpoint(ENDPOINT_NAME)


@pytest.fixture(scope="module")
def reference_df():
    """Reference-compounds table."""
    ref_df = PublicData().get(REFERENCE_DATASET)
    assert ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return ref_df


@pytest.fixture(scope="module")
def result_df(endpoint, reference_df):
    """Async inference on all reference compounds with expected-range cols attached."""
    payload = reference_df[["id", "smiles"]].copy()
    pred = endpoint.inference(payload)
    return pred.merge(
        reference_df.drop(columns=["smiles"]),
        on="id",
        how="left",
        suffixes=("", "_expected"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_endpoint_exists(endpoint):
    """Async endpoint is deployed and healthy."""
    assert endpoint.exists(), f"Endpoint '{ENDPOINT_NAME}' does not exist"
    print(f"Endpoint: {endpoint.name}")


def test_reference_compounds_loadable(reference_df):
    """Reference-compounds dataset is accessible and non-empty."""
    assert len(reference_df) > 0
    assert {"id", "name", "smiles"}.issubset(reference_df.columns)
    print(f"Loaded {len(reference_df)} reference compounds")


def test_all_reference_compounds_succeed(result_df):
    """Every reference compound returns desc3d_status == 'ok'."""
    bad = result_df[result_df["desc3d_status"] != "ok"]
    if not bad.empty:
        print(bad[["name", "smiles", "desc3d_status"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} reference compounds did not compute successfully"


def test_adaptive_conformer_count(result_df):
    """Endpoint should use adaptive conformer counts (50 / 300 / 500)."""
    requested = result_df["desc3d_confs_requested"].dropna().unique().tolist()
    # Adaptive tiers per mol_descriptors_3d.adaptive_n_conformers
    allowed = {50, 300, 500}
    bad = [n for n in requested if int(n) not in allowed]
    assert not bad, f"Unexpected conformer counts requested: {bad} (allowed: {sorted(allowed)})"
    print(f"Conformer counts used: {sorted(int(n) for n in requested)}")


def test_energy_method_is_xtb(result_df):
    """Every reference compound is Boltzmann-weighted with GFN2-xTB energies.

    This is the guard against the silent xTB→MMFF fallback: if the deployed
    image is missing tblite or pins a stale workbench, get_conformer_energies
    quietly falls back to MMFF and features look valid but are wrongly
    weighted. The curated reference compounds all score cleanly under xTB, so
    anything other than 'GFN2-xTB' here means the energy path is broken in the
    deployed endpoint.
    """
    ok = result_df[result_df["desc3d_status"] == "ok"]
    bad = ok[ok["desc3d_energy_method"] != "GFN2-xTB"]
    if not bad.empty:
        print(bad[["name", "smiles", "desc3d_energy_method"]].to_string(index=False))
    assert bad.empty, (
        f"{len(bad)} compounds not weighted by GFN2-xTB "
        f"(fell back to: {sorted(bad['desc3d_energy_method'].unique())}) — "
        "check tblite is installed and WORKBENCH_VERSION is current in the deployed image"
    )


def test_compute_time_reported(result_df):
    """desc3d_compute_time_s is populated and positive; report the distribution.

    Not a threshold assertion (wall time is instance- and molecule-dependent),
    but surfaces the xTB cost so regressions/baselines are visible in CI logs.
    """
    ok = result_df[result_df["desc3d_status"] == "ok"].copy()
    times = ok["desc3d_compute_time_s"].astype(float)
    assert (times > 0).all(), "desc3d_compute_time_s should be positive for successful rows"

    slowest = ok.nlargest(5, "desc3d_compute_time_s")[["name", "desc3d_conf_count", "desc3d_compute_time_s"]]
    print(f"compute_time_s — median={times.median():.2f}s max={times.max():.2f}s")
    print("slowest compounds:")
    print(slowest.to_string(index=False))


def test_all_features_finite(result_df):
    """All 74 feature columns present and non-NaN for every reference compound."""
    feature_names = get_3d_feature_names()
    missing = [c for c in feature_names if c not in result_df.columns]
    assert not missing, f"Missing feature columns: {missing[:5]}"

    nan_rows = result_df[result_df[feature_names].isna().any(axis=1)]
    if not nan_rows.empty:
        for _, row in nan_rows.iterrows():
            bad_cols = [c for c in feature_names if pd.isna(row[c])]
            head = ", ".join(bad_cols[:5]) + ("..." if len(bad_cols) > 5 else "")
            print(f"  {row['name']}: NaN in [{head}]")
    assert nan_rows.empty, f"{len(nan_rows)} compounds have NaN feature values"


def test_pmi_ordering(result_df):
    """Principal moments of inertia satisfy PMI1 <= PMI2 <= PMI3."""
    ordered = (result_df["pmi1"] <= result_df["pmi2"]) & (result_df["pmi2"] <= result_df["pmi3"])
    bad = result_df[~ordered]
    if not bad.empty:
        print(bad[["name", "pmi1", "pmi2", "pmi3"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} compounds violate PMI ordering"


def test_npr_bounds(result_df):
    """NPR1, NPR2 in [0, 1], NPR1 <= NPR2, NPR1 + NPR2 >= 1 (mass-distribution invariant)."""
    n1, n2 = result_df["npr1"], result_df["npr2"]
    valid = (n1 >= 0) & (n1 <= 1) & (n2 >= 0) & (n2 <= 1) & (n1 <= n2) & ((n1 + n2) >= 0.99)
    bad = result_df[~valid]
    if not bad.empty:
        print(bad[["name", "npr1", "npr2"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} compounds violate NPR bounds"


def test_geometric_positivity(result_df):
    """Geometric descriptors are strictly positive (or non-negative, as appropriate)."""
    for col in ["radius_of_gyration", "pharm3d_molecular_volume", "pharm3d_molecular_axis"]:
        bad = result_df[result_df[col] <= 0]
        if not bad.empty:
            print(bad[["name", col]].to_string(index=False))
        assert bad.empty, f"{col} not strictly positive for {len(bad)} compounds"

    for col in ["asphericity", "spherocity_index"]:
        bad = result_df[result_df[col] < 0]
        if not bad.empty:
            print(bad[["name", col]].to_string(index=False))
        assert bad.empty, f"{col} has negative values for {len(bad)} compounds"


def test_expected_ranges(result_df):
    """Every compound-specific expected range (where set) is satisfied."""

    checks = [
        ("npr1", "npr1_min", "npr1_max"),
        ("npr2", "npr2_min", "npr2_max"),
        ("pharm3d_imhb_potential", "imhb_min", "imhb_max"),
        ("pharm3d_molecular_axis", "axis_min", "axis_max"),
        ("pharm3d_nitrogen_span", "nitrogen_span_min", "nitrogen_span_max"),
    ]

    failures = []
    for feature, lo_col, hi_col in checks:
        for _, row in result_df.iterrows():
            lo, hi, val = row[lo_col], row[hi_col], row[feature]
            if pd.isna(lo) or pd.isna(hi):
                continue
            if not (lo <= val <= hi):
                failures.append((row["name"], feature, float(val), float(lo), float(hi)))

    for name, feat, val, lo, hi in failures:
        print(f"  {name}: {feat}={val:.3f} outside [{lo}, {hi}]")
    assert not failures, f"{len(failures)} expected-range assertions failed"


def test_undefined_chiral_centers_match_expected(result_df):
    """Each reference compound's undefined_chiral_centers matches the expected value."""
    failures = []
    for _, row in result_df.iterrows():
        expected = row.get("undefined_chiral_centers_expected")
        if pd.isna(expected):
            continue
        actual = row["undefined_chiral_centers"]
        if int(actual) != int(expected):
            failures.append((row["name"], int(actual), int(expected)))

    for name, actual, expected in failures:
        print(f"  {name}: undefined_chiral_centers={actual}, expected={expected}")
    assert not failures, f"{len(failures)} compounds have wrong undefined_chiral_centers count"


def test_stereo_preserved_matches_expected(result_df):
    """Each reference compound's desc3d_stereo_preserved matches the expected value."""
    failures = []
    for _, row in result_df.iterrows():
        expected = row.get("stereo_preserved_expected")
        if pd.isna(expected):
            continue
        actual = row["desc3d_stereo_preserved"]
        if pd.isna(actual) or bool(actual) != bool(expected):
            failures.append((row["name"], actual, expected))

    for name, actual, expected in failures:
        print(f"  {name}: desc3d_stereo_preserved={actual}, expected={expected}")
    assert not failures, f"{len(failures)} compounds have wrong desc3d_stereo_preserved value"


if __name__ == "__main__":
    # Build the fixture objects directly for standalone runs.
    ep = Endpoint(ENDPOINT_NAME)
    ref_df = PublicData().get(REFERENCE_DATASET)
    assert ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    res_df = ep.inference(ref_df[["id", "smiles"]].copy()).merge(
        ref_df.drop(columns=["smiles"]),
        on="id",
        how="left",
        suffixes=("", "_expected"),
    )

    test_endpoint_exists(ep)
    test_reference_compounds_loadable(ref_df)
    test_all_reference_compounds_succeed(res_df)
    test_adaptive_conformer_count(res_df)
    test_energy_method_is_xtb(res_df)
    test_compute_time_reported(res_df)
    test_all_features_finite(res_df)
    test_pmi_ordering(res_df)
    test_npr_bounds(res_df)
    test_geometric_positivity(res_df)
    test_expected_ranges(res_df)
    test_undefined_chiral_centers_match_expected(res_df)
    test_stereo_preserved_matches_expected(res_df)
    print("\nAll Boltzmann 3D endpoint validation tests passed!")
