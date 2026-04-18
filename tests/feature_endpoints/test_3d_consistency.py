"""Cross-endpoint consistency tests for the two 3D descriptor endpoints.

Compares the outputs of:
    - smiles-to-3d-descriptors-v1    (realtime, 10 conformers)
    - smiles-to-3d-boltzmann-v1      (async,  50-300 adaptive conformers)

Both endpoints share the same descriptor computation and Boltzmann-weighted
aggregation code, so for any given compound the feature values should agree
within tolerance. Perfect agreement is not expected because Boltzmann mode
samples the energy basin more densely, which can shift flexible-molecule
values by 10-20%.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/reference_compounds_3d

Populated by: scripts/admin/populate_3d_reference_compounds.py
"""

import pandas as pd

from workbench.api import AsyncEndpoint, Endpoint, PublicData
from workbench.utils.chem_utils.mol_descriptors_3d import get_3d_feature_names

FAST_ENDPOINT = "smiles-to-3d-descriptors-v1"
BOLTZMANN_ENDPOINT = "smiles-to-3d-boltzmann-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/reference_compounds_3d"

fast_endpoint = Endpoint(FAST_ENDPOINT)
boltzmann_endpoint = AsyncEndpoint(BOLTZMANN_ENDPOINT)

# ---------------------------------------------------------------------------
# Per-feature tolerances for cross-endpoint comparison.
#
#   feature passes if:
#       abs(fast - boltzmann) <= max(abs_tol, rel_tol * max(|fast|, |boltzmann|))
#
# Conformer-ensemble statistics (conf_energy_*, conformational_flexibility,
# desc3d_conf_count) intentionally differ between modes — fast mode samples
# at most 10 conformers, Boltzmann mode samples 50-300 — so they're excluded
# from the comparison.
# ---------------------------------------------------------------------------

# Descriptors that differ by design between the two modes
SKIPPED_FEATURES = {
    "conf_energy_min",
    "conf_energy_range",
    "conf_energy_std",
    "conformational_flexibility",
}

# Descriptors bounded to [0, 1] (or similar small range)
BOUNDED_FEATURES = {
    "npr1",
    "npr2",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "spherocity_index",
}

# Default tolerances for the remaining descriptors (Mordred 3D, pharmacophore,
# radius_of_gyration, PMIs). abs_tol catches noise for near-zero values;
# rel_tol catches divergence for large values.
DEFAULT_ABS_TOL = 0.5
DEFAULT_REL_TOL = 0.25

# Tighter tolerance on bounded descriptors (they can't drift far)
BOUNDED_ABS_TOL = 0.15
BOUNDED_REL_TOL = 0.15


def _tolerance_for(feature: str) -> tuple:
    """Return (abs_tol, rel_tol) for a feature, or None to skip."""
    if feature in SKIPPED_FEATURES:
        return None
    if feature in BOUNDED_FEATURES:
        return (BOUNDED_ABS_TOL, BOUNDED_REL_TOL)
    if feature == "pharm3d_imhb_potential":
        # Integer count averaged over conformers — can differ by ~1 between modes
        return (1.5, 0.0)
    return (DEFAULT_ABS_TOL, DEFAULT_REL_TOL)


# ---------------------------------------------------------------------------
# Cached inference results
# ---------------------------------------------------------------------------

_ref_df = None
_fast_df = None
_bolt_df = None


def _get_reference_df() -> pd.DataFrame:
    global _ref_df
    if _ref_df is None:
        _ref_df = PublicData().get(REFERENCE_DATASET)
        assert _ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return _ref_df


def _get_fast_df() -> pd.DataFrame:
    global _fast_df
    if _fast_df is None:
        ref_df = _get_reference_df()
        _fast_df = fast_endpoint.inference(ref_df[["id", "smiles"]].copy())
    return _fast_df


def _get_boltzmann_df() -> pd.DataFrame:
    global _bolt_df
    if _bolt_df is None:
        ref_df = _get_reference_df()
        _bolt_df = boltzmann_endpoint.inference(ref_df[["id", "smiles"]].copy())
    return _bolt_df


def _joined() -> pd.DataFrame:
    """Return reference rows joined with both endpoints' outputs (suffixed _fast / _bolt)."""
    ref = _get_reference_df()[["id", "name"]]
    fast = _get_fast_df().add_suffix("_fast").rename(columns={"id_fast": "id"})
    bolt = _get_boltzmann_df().add_suffix("_bolt").rename(columns={"id_bolt": "id"})
    return ref.merge(fast, on="id", how="inner").merge(bolt, on="id", how="inner")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_both_endpoints_exist():
    assert fast_endpoint.exists(), f"Endpoint '{FAST_ENDPOINT}' does not exist"
    assert boltzmann_endpoint.exists(), f"Endpoint '{BOLTZMANN_ENDPOINT}' does not exist"


def test_both_endpoints_succeed_on_all_compounds():
    """Every reference compound computes successfully on both endpoints."""
    joined = _joined()
    bad_fast = joined[joined["desc3d_status_fast"] != "ok"]
    bad_bolt = joined[joined["desc3d_status_bolt"] != "ok"]
    if not bad_fast.empty:
        print("Fast endpoint failures:")
        print(bad_fast[["name", "desc3d_status_fast"]].to_string(index=False))
    if not bad_bolt.empty:
        print("Boltzmann endpoint failures:")
        print(bad_bolt[["name", "desc3d_status_bolt"]].to_string(index=False))
    assert bad_fast.empty and bad_bolt.empty


def test_mode_labels_differ():
    """Fast endpoint reports mode='fast', Boltzmann reports mode='boltzmann'."""
    joined = _joined()
    fast_modes = set(joined["desc3d_mode_fast"].unique())
    bolt_modes = set(joined["desc3d_mode_bolt"].unique())
    assert fast_modes == {"fast"}, f"Fast endpoint modes: {fast_modes}"
    assert bolt_modes == {"boltzmann"}, f"Boltzmann endpoint modes: {bolt_modes}"


def test_boltzmann_samples_at_least_as_many_conformers():
    """For every compound, Boltzmann conformer count >= fast conformer count."""
    joined = _joined()
    bad = joined[joined["desc3d_conf_count_bolt"] < joined["desc3d_conf_count_fast"]]
    if not bad.empty:
        print(bad[["name", "desc3d_conf_count_fast", "desc3d_conf_count_bolt"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} compounds have Boltzmann conf_count < fast conf_count"


def test_feature_agreement():
    """Every non-excluded feature agrees between the two endpoints within tolerance."""
    joined = _joined()
    feature_names = get_3d_feature_names()

    failures = []
    checked = 0

    for feature in feature_names:
        tol = _tolerance_for(feature)
        if tol is None:
            continue
        abs_tol, rel_tol = tol

        fast_col = f"{feature}_fast"
        bolt_col = f"{feature}_bolt"
        if fast_col not in joined.columns or bolt_col not in joined.columns:
            failures.append(("<all>", feature, "missing_column", None, None, None))
            continue

        checked += 1
        for _, row in joined.iterrows():
            fv, bv = row[fast_col], row[bolt_col]
            if pd.isna(fv) or pd.isna(bv):
                # NaN feature values are covered by the individual endpoint tests;
                # skip here to avoid double-reporting.
                continue
            diff = abs(fv - bv)
            threshold = max(abs_tol, rel_tol * max(abs(fv), abs(bv)))
            if diff > threshold:
                failures.append((row["name"], feature, "diff", float(fv), float(bv), float(threshold)))

    if failures:
        # Group failures by compound for readable output
        print(f"Feature agreement failures ({len(failures)} across {checked} checked features):")
        failures_df = pd.DataFrame(
            failures, columns=["compound", "feature", "reason", "fast", "boltzmann", "threshold"]
        )
        with pd.option_context("display.max_rows", 100, "display.width", 160):
            print(failures_df.to_string(index=False))

    assert not failures, f"{len(failures)} feature-agreement checks failed"


def test_shape_descriptors_strongly_agree():
    """Rigid-molecule shape descriptors should agree tightly (sanity subset).

    For benzene and adamantane — rigid molecules where every conformer is
    essentially identical — NPR values must agree within 0.05 regardless of
    conformer count. Tight threshold catches systematic divergence that the
    full-feature test's 0.15 tolerance would miss.
    """
    joined = _joined()
    rigid = joined[joined["name"].isin(["benzene", "adamantane"])]
    assert not rigid.empty, "Expected rigid molecules (benzene, adamantane) in reference set"

    failures = []
    for _, row in rigid.iterrows():
        for feature in ["npr1", "npr2", "asphericity", "spherocity_index"]:
            diff = abs(row[f"{feature}_fast"] - row[f"{feature}_bolt"])
            if diff > 0.05:
                failures.append((row["name"], feature, row[f"{feature}_fast"], row[f"{feature}_bolt"], diff))

    for name, feat, fv, bv, diff in failures:
        print(f"  {name}: {feat} fast={fv:.3f} bolt={bv:.3f} diff={diff:.3f}")
    assert not failures, f"{len(failures)} rigid-molecule shape descriptors diverge > 0.05"


if __name__ == "__main__":
    test_both_endpoints_exist()
    test_both_endpoints_succeed_on_all_compounds()
    test_mode_labels_differ()
    test_boltzmann_samples_at_least_as_many_conformers()
    test_feature_agreement()
    test_shape_descriptors_strongly_agree()
    print("\nAll 3D endpoint consistency tests passed!")
