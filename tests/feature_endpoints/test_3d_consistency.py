"""Cross-endpoint consistency tests for the two 3D descriptor endpoints.

Compares the outputs of:
    - smiles-to-3d-fast-v1    (realtime, 10 conformers)
    - smiles-to-3d-full-v1      (async,  50-300 adaptive conformers)

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
import pytest

from workbench.api import Endpoint, PublicData
from workbench.utils.chem_utils.mol_descriptors_3d import get_3d_feature_names

# Heavy validation study (~6 min): medium tier, excluded from quick runs
pytestmark = pytest.mark.medium

FAST_ENDPOINT = "smiles-to-3d-fast-v1"
FULL_ENDPOINT = "smiles-to-3d-full-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/reference_compounds_3d"

# ---------------------------------------------------------------------------
# Per-feature tolerances for cross-endpoint comparison.
#
#   feature passes if:
#       abs(fast - full) <= max(abs_tol, rel_tol * max(|fast|, |full|))
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
# Fixtures: module-scoped so AWS calls happen once per file, at run time
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_endpoint():
    return Endpoint(FAST_ENDPOINT)


@pytest.fixture(scope="module")
def full_endpoint():
    return Endpoint(FULL_ENDPOINT)


@pytest.fixture(scope="module")
def reference_df():
    df = PublicData().get(REFERENCE_DATASET)
    assert df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return df


@pytest.fixture(scope="module")
def fast_df(fast_endpoint, reference_df):
    return fast_endpoint.inference(reference_df[["id", "smiles"]].copy())


@pytest.fixture(scope="module")
def full_df(full_endpoint, reference_df):
    return full_endpoint.inference(reference_df[["id", "smiles"]].copy())


@pytest.fixture(scope="module")
def joined(reference_df, fast_df, full_df):
    """Reference rows joined with both endpoints' outputs (suffixed _fast / _bolt)."""
    ref = reference_df[["id", "name"]]
    fast = fast_df.add_suffix("_fast").rename(columns={"id_fast": "id"})
    bolt = full_df.add_suffix("_bolt").rename(columns={"id_bolt": "id"})
    return ref.merge(fast, on="id", how="inner").merge(bolt, on="id", how="inner")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_both_endpoints_exist(fast_endpoint, full_endpoint):
    assert fast_endpoint.exists(), f"Endpoint '{FAST_ENDPOINT}' does not exist"
    assert full_endpoint.exists(), f"Endpoint '{FULL_ENDPOINT}' does not exist"


def test_both_endpoints_succeed_on_all_compounds(joined):
    """Every reference compound computes successfully on both endpoints."""
    bad_fast = joined[joined["desc3d_status_fast"] != "ok"]
    bad_bolt = joined[joined["desc3d_status_bolt"] != "ok"]
    if not bad_fast.empty:
        print("Fast endpoint failures:")
        print(bad_fast[["name", "desc3d_status_fast"]].to_string(index=False))
    if not bad_bolt.empty:
        print("Boltzmann endpoint failures:")
        print(bad_bolt[["name", "desc3d_status_bolt"]].to_string(index=False))
    assert bad_fast.empty and bad_bolt.empty


def test_mode_labels_differ(joined):
    """Fast endpoint reports mode='fast', Boltzmann reports mode='full'."""
    fast_modes = set(joined["desc3d_mode_fast"].unique())
    bolt_modes = set(joined["desc3d_mode_bolt"].unique())
    assert fast_modes == {"fast"}, f"Fast endpoint modes: {fast_modes}"
    assert bolt_modes == {"full"}, f"Boltzmann endpoint modes: {bolt_modes}"


def test_full_samples_at_least_as_many_conformers(joined):
    """For every compound, Boltzmann conformer count >= fast conformer count."""
    bad = joined[joined["desc3d_conf_count_bolt"] < joined["desc3d_conf_count_fast"]]
    if not bad.empty:
        print(bad[["name", "desc3d_conf_count_fast", "desc3d_conf_count_bolt"]].to_string(index=False))
    assert bad.empty, f"{len(bad)} compounds have Boltzmann conf_count < fast conf_count"


def test_feature_agreement(joined):
    """Every non-excluded feature agrees between the two endpoints within tolerance."""
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
        failures_df = pd.DataFrame(failures, columns=["compound", "feature", "reason", "fast", "full", "threshold"])
        with pd.option_context("display.max_rows", 100, "display.width", 160):
            print(failures_df.to_string(index=False))

    assert not failures, f"{len(failures)} feature-agreement checks failed"


def test_shape_descriptors_strongly_agree(joined):
    """Rigid-molecule shape descriptors should agree tightly (sanity subset).

    For benzene and adamantane — rigid molecules where every conformer is
    essentially identical — NPR values must agree within 0.05 regardless of
    conformer count. Tight threshold catches systematic divergence that the
    full-feature test's 0.15 tolerance would miss.
    """
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
    # Construct artifacts and data directly, then pass to tests explicitly
    fast_ep = Endpoint(FAST_ENDPOINT)
    full_ep = Endpoint(FULL_ENDPOINT)
    ref = PublicData().get(REFERENCE_DATASET)
    fast = fast_ep.inference(ref[["id", "smiles"]].copy())
    bolt = full_ep.inference(ref[["id", "smiles"]].copy())
    joined_df = (
        ref[["id", "name"]]
        .merge(fast.add_suffix("_fast").rename(columns={"id_fast": "id"}), on="id", how="inner")
        .merge(bolt.add_suffix("_bolt").rename(columns={"id_bolt": "id"}), on="id", how="inner")
    )

    test_both_endpoints_exist(fast_ep, full_ep)
    test_both_endpoints_succeed_on_all_compounds(joined_df)
    test_mode_labels_differ(joined_df)
    test_full_samples_at_least_as_many_conformers(joined_df)
    test_feature_agreement(joined_df)
    test_shape_descriptors_strongly_agree(joined_df)
    print("\nAll 3D endpoint consistency tests passed!")
