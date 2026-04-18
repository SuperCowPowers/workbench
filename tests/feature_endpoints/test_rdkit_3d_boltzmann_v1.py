"""Validation tests for the async Boltzmann 3D descriptor endpoint (smiles-to-3d-boltzmann-v1).

Pulls the same curated reference compounds used by test_rdkit_3d_v1.py and runs
them through the async (Boltzmann-ensemble) endpoint. Validates the same
invariants and compound-specific expected ranges — the two endpoints are
expected to produce numerically similar (though not identical) results because
Boltzmann mode uses adaptive conformer counts (50-300) vs the fast mode's 10.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/reference_compounds_3d

Populated by: scripts/admin/populate_3d_reference_compounds.py
"""

import pandas as pd

from workbench.api import AsyncEndpoint, PublicData
from workbench.utils.chem_utils.mol_descriptors_3d import get_3d_feature_names

ENDPOINT_NAME = "smiles-to-3d-boltzmann-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/reference_compounds_3d"

endpoint = AsyncEndpoint(ENDPOINT_NAME)

# Module-level caches — Boltzmann inference is expensive (minutes for the full
# reference set) so we run it exactly once and reuse the result for every test.
_ref_df = None
_result_df = None


def _get_reference_df() -> pd.DataFrame:
    """Load the reference-compounds table (cached)."""
    global _ref_df
    if _ref_df is None:
        _ref_df = PublicData().get(REFERENCE_DATASET)
        assert _ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return _ref_df


def _get_result() -> pd.DataFrame:
    """Run async inference on all reference compounds (cached)."""
    global _result_df
    if _result_df is None:
        ref_df = _get_reference_df()
        payload = ref_df[["id", "smiles"]].copy()
        pred = endpoint.inference(payload)
        _result_df = pred.merge(
            ref_df.drop(columns=["smiles"]),
            on="id",
            how="left",
            suffixes=("", "_expected"),
        )
    return _result_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_endpoint_exists():
    """Async endpoint is deployed and healthy."""
    assert endpoint.exists(), f"Endpoint '{ENDPOINT_NAME}' does not exist"
    print(f"AsyncEndpoint: {endpoint.name}")


def test_reference_compounds_loadable():
    """Reference-compounds dataset is accessible and non-empty."""
    ref_df = _get_reference_df()
    assert len(ref_df) > 0
    assert {"id", "name", "smiles"}.issubset(ref_df.columns)
    print(f"Loaded {len(ref_df)} reference compounds")


def test_all_reference_compounds_succeed():
    """Every reference compound returns desc3d_status == 'ok'."""
    result = _get_result()
    bad = result[result["desc3d_status"] != "ok"]
    if not bad.empty:
        print(bad[["name", "smiles", "desc3d_status"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} reference compounds did not compute successfully"


def test_boltzmann_mode_declared():
    """Every row reports desc3d_mode == 'boltzmann'."""
    result = _get_result()
    modes = result["desc3d_mode"].unique().tolist()
    assert modes == ["boltzmann"], f"Expected desc3d_mode='boltzmann', got {modes}"


def test_adaptive_conformer_count():
    """Boltzmann mode should use adaptive conformer counts (50 / 200 / 300)."""
    result = _get_result()
    requested = result["desc3d_confs_requested"].dropna().unique().tolist()
    # Adaptive tiers per mol_descriptors_3d.adaptive_n_conformers
    allowed = {50, 200, 300}
    bad = [n for n in requested if int(n) not in allowed]
    assert not bad, f"Unexpected conformer counts requested: {bad} (allowed: {sorted(allowed)})"
    print(f"Conformer counts used: {sorted(int(n) for n in requested)}")


def test_all_features_finite():
    """All 74 feature columns present and non-NaN for every reference compound."""
    result = _get_result()
    feature_names = get_3d_feature_names()
    missing = [c for c in feature_names if c not in result.columns]
    assert not missing, f"Missing feature columns: {missing[:5]}"

    nan_rows = result[result[feature_names].isna().any(axis=1)]
    if not nan_rows.empty:
        for _, row in nan_rows.iterrows():
            bad_cols = [c for c in feature_names if pd.isna(row[c])]
            head = ", ".join(bad_cols[:5]) + ("..." if len(bad_cols) > 5 else "")
            print(f"  {row['name']}: NaN in [{head}]")
    assert nan_rows.empty, f"{len(nan_rows)} compounds have NaN feature values"


def test_pmi_ordering():
    """Principal moments of inertia satisfy PMI1 <= PMI2 <= PMI3."""
    result = _get_result()
    ordered = (result["pmi1"] <= result["pmi2"]) & (result["pmi2"] <= result["pmi3"])
    bad = result[~ordered]
    if not bad.empty:
        print(bad[["name", "pmi1", "pmi2", "pmi3"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} compounds violate PMI ordering"


def test_npr_bounds():
    """NPR1, NPR2 in [0, 1], NPR1 <= NPR2, NPR1 + NPR2 >= 1 (mass-distribution invariant)."""
    result = _get_result()
    n1, n2 = result["npr1"], result["npr2"]
    valid = (n1 >= 0) & (n1 <= 1) & (n2 >= 0) & (n2 <= 1) & (n1 <= n2) & ((n1 + n2) >= 0.99)
    bad = result[~valid]
    if not bad.empty:
        print(bad[["name", "npr1", "npr2"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} compounds violate NPR bounds"


def test_geometric_positivity():
    """Geometric descriptors are strictly positive (or non-negative, as appropriate)."""
    result = _get_result()
    for col in ["radius_of_gyration", "pharm3d_molecular_volume", "pharm3d_molecular_axis"]:
        bad = result[result[col] <= 0]
        if not bad.empty:
            print(bad[["name", col]].to_string(index=False))
        assert bad.empty, f"{col} not strictly positive for {len(bad)} compounds"

    for col in ["asphericity", "spherocity_index"]:
        bad = result[result[col] < 0]
        if not bad.empty:
            print(bad[["name", col]].to_string(index=False))
        assert bad.empty, f"{col} has negative values for {len(bad)} compounds"


def test_expected_ranges():
    """Every compound-specific expected range (where set) is satisfied."""
    result = _get_result()

    checks = [
        ("npr1", "npr1_min", "npr1_max"),
        ("npr2", "npr2_min", "npr2_max"),
        ("pharm3d_imhb_potential", "imhb_min", "imhb_max"),
        ("pharm3d_molecular_axis", "axis_min", "axis_max"),
        ("pharm3d_nitrogen_span", "nitrogen_span_min", "nitrogen_span_max"),
    ]

    failures = []
    for feature, lo_col, hi_col in checks:
        for _, row in result.iterrows():
            lo, hi, val = row[lo_col], row[hi_col], row[feature]
            if pd.isna(lo) or pd.isna(hi):
                continue
            if not (lo <= val <= hi):
                failures.append((row["name"], feature, float(val), float(lo), float(hi)))

    for name, feat, val, lo, hi in failures:
        print(f"  {name}: {feat}={val:.3f} outside [{lo}, {hi}]")
    assert not failures, f"{len(failures)} expected-range assertions failed"


def test_undefined_chiral_centers_match_expected():
    """Each reference compound's undefined_chiral_centers matches the expected value."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
        expected = row.get("undefined_chiral_centers_expected")
        if pd.isna(expected):
            continue
        actual = row["undefined_chiral_centers"]
        if int(actual) != int(expected):
            failures.append((row["name"], int(actual), int(expected)))

    for name, actual, expected in failures:
        print(f"  {name}: undefined_chiral_centers={actual}, expected={expected}")
    assert not failures, f"{len(failures)} compounds have wrong undefined_chiral_centers count"


def test_stereo_preserved_matches_expected():
    """Each reference compound's desc3d_stereo_preserved matches the expected value."""
    result = _get_result()
    failures = []
    for _, row in result.iterrows():
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
    test_endpoint_exists()
    test_reference_compounds_loadable()
    test_all_reference_compounds_succeed()
    test_boltzmann_mode_declared()
    test_adaptive_conformer_count()
    test_all_features_finite()
    test_pmi_ordering()
    test_npr_bounds()
    test_geometric_positivity()
    test_expected_ranges()
    test_undefined_chiral_centers_match_expected()
    test_stereo_preserved_matches_expected()
    print("\nAll Boltzmann 3D endpoint validation tests passed!")
