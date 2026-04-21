"""Flexibility-driven fast-vs-Boltzmann comparison for the 3D endpoints.

Pulls a reference set of compounds spanning the rotatable-bond spectrum
(0 → 19 rot bonds) and reports the per-compound divergence between the
fast endpoint (10 conformers) and the Boltzmann async endpoint (50-300
adaptive conformers) on a curated subset of shape/size descriptors.

What it proves:
    - Rigid compounds (0 rot bonds): both endpoints must produce nearly
      identical values (shape descriptors within 0.05 absolute). This is
      a regression gate — any divergence means the two endpoints are no
      longer computing the same underlying quantities.
    - Semi-rigid → very flexible: no strict gate, just reporting. Shows
      how much more thorough conformer sampling (Boltzmann) actually
      shifts ensemble-averaged descriptors for flexible molecules.

The reference compounds are loaded from:
    s3://workbench-public-data/comp_chem/reference_compounds/flexibility

Populated by: scripts/admin/populate_flexibility_reference_compounds.py
"""

import pandas as pd

from workbench.api import AsyncEndpoint, Endpoint, PublicData

FAST_ENDPOINT = "smiles-to-3d-fast-v1"
FULL_ENDPOINT = "smiles-to-3d-full-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/flexibility"

# Features to report on. Shape descriptors (NPR, asphericity) are bounded
# [0, 1] and the most directly conformer-dependent. Size descriptors
# (radius_of_gyration, molecular_axis, molecular_volume) scale with the
# molecule and are sensitive to extended-vs-compact conformer sampling.
# Elongation is derived (axis / volume^(1/3)) and amplifies divergence.
REPORTED_FEATURES = [
    "npr1",
    "npr2",
    "asphericity",
    "radius_of_gyration",
    "pharm3d_molecular_axis",
    "pharm3d_molecular_volume",
    "pharm3d_elongation",
]

# Strict-agreement tolerance for rigid compounds. Shape descriptors must
# agree within this absolute delta; size descriptors within this relative
# fraction. Rigid = every conformer is essentially identical, so these
# bounds should be easy.
RIGID_ABS_TOL_BOUNDED = 0.05
RIGID_REL_TOL_SIZE = 0.05

BOUNDED_FEATURES = {"npr1", "npr2", "asphericity"}

# Order used when printing class-grouped sections in the report.
CLASS_ORDER = ["rigid", "semi_rigid", "flexible", "very_flexible"]

fast_endpoint = Endpoint(FAST_ENDPOINT)
full_endpoint = AsyncEndpoint(FULL_ENDPOINT)

# ---------------------------------------------------------------------------
# Cached inference results
# ---------------------------------------------------------------------------

_ref_df = None
_joined_df = None


def _get_reference_df() -> pd.DataFrame:
    global _ref_df
    if _ref_df is None:
        _ref_df = PublicData().get(REFERENCE_DATASET)
        assert _ref_df is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    return _ref_df


def _get_joined_df() -> pd.DataFrame:
    """Reference rows joined with both endpoints' outputs (suffixed _fast / _bolt)."""
    global _joined_df
    if _joined_df is None:
        ref_df = _get_reference_df()
        payload = ref_df[["id", "smiles"]].copy()
        fast = fast_endpoint.inference(payload).add_suffix("_fast").rename(columns={"id_fast": "id"})
        bolt = full_endpoint.inference(payload).add_suffix("_bolt").rename(columns={"id_bolt": "id"})
        _joined_df = (
            ref_df[["id", "name", "rot_bonds", "flexibility_class", "notes"]]
            .merge(fast, on="id", how="inner")
            .merge(bolt, on="id", how="inner")
        )
    return _joined_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_delta(fast_val: float, bolt_val: float) -> str:
    """Pretty-print fast vs full values and their absolute + relative diff."""
    if pd.isna(fast_val) or pd.isna(bolt_val):
        return f"fast={fast_val}  bolt={bolt_val}  (NaN)"
    abs_diff = abs(fast_val - bolt_val)
    scale = max(abs(fast_val), abs(bolt_val), 1e-9)
    rel_pct = 100.0 * abs_diff / scale
    return f"fast={fast_val:8.3f}  bolt={bolt_val:8.3f}  Δ={abs_diff:7.3f}  ({rel_pct:5.1f}%)"


def _print_report(joined: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print("3D descriptor flexibility comparison — fast vs Boltzmann")
    print("=" * 90)

    for cls in CLASS_ORDER:
        subset = joined[joined["flexibility_class"] == cls]
        if subset.empty:
            continue
        print()
        print(f"--- {cls.upper()} ({len(subset)} compound{'s' if len(subset) != 1 else ''}) ---")
        for _, row in subset.iterrows():
            print(f"  {row['name']} (rot_bonds={int(row['rot_bonds'])})")
            for feat in REPORTED_FEATURES:
                fast_val = row.get(f"{feat}_fast")
                bolt_val = row.get(f"{feat}_bolt")
                print(f"    {feat:28} {_format_delta(fast_val, bolt_val)}")
    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_both_endpoints_exist():
    assert fast_endpoint.exists()
    assert full_endpoint.exists()


def test_reference_compounds_loadable():
    ref_df = _get_reference_df()
    assert len(ref_df) > 0
    assert {"id", "name", "smiles", "rot_bonds", "flexibility_class"}.issubset(ref_df.columns)


def test_both_endpoints_succeed_on_all_compounds():
    joined = _get_joined_df()
    bad_fast = joined[joined["desc3d_status_fast"] != "ok"]
    bad_bolt = joined[joined["desc3d_status_bolt"] != "ok"]
    if not bad_fast.empty:
        print("Fast endpoint failures:")
        print(bad_fast[["name", "desc3d_status_fast"]].to_string(index=False))
    if not bad_bolt.empty:
        print("Boltzmann endpoint failures:")
        print(bad_bolt[["name", "desc3d_status_bolt"]].to_string(index=False))
    assert bad_fast.empty and bad_bolt.empty


def test_print_full_comparison_report():
    """Not a gate — just prints the per-compound delta report for human review."""
    _print_report(_get_joined_df())


def test_rigid_compounds_agree_tightly():
    """Regression gate: rigid-class compounds (0 rot bonds) must produce
    nearly identical feature values on both endpoints. Tight agreement here
    proves the two endpoints implement the same descriptor math — any
    divergence means something has drifted.
    """
    joined = _get_joined_df()
    rigid = joined[joined["flexibility_class"] == "rigid"]
    assert not rigid.empty, "Expected rigid-class compounds in the reference set"

    failures = []
    for _, row in rigid.iterrows():
        for feat in REPORTED_FEATURES:
            fast_val = row[f"{feat}_fast"]
            bolt_val = row[f"{feat}_bolt"]
            if pd.isna(fast_val) or pd.isna(bolt_val):
                failures.append((row["name"], feat, "NaN", fast_val, bolt_val, None))
                continue
            abs_diff = abs(fast_val - bolt_val)
            if feat in BOUNDED_FEATURES:
                tol = RIGID_ABS_TOL_BOUNDED
                if abs_diff > tol:
                    failures.append((row["name"], feat, "abs", fast_val, bolt_val, tol))
            else:
                scale = max(abs(fast_val), abs(bolt_val), 1e-9)
                rel_diff = abs_diff / scale
                if rel_diff > RIGID_REL_TOL_SIZE:
                    failures.append((row["name"], feat, "rel", fast_val, bolt_val, RIGID_REL_TOL_SIZE))

    for name, feat, kind, fv, bv, tol in failures:
        print(f"  {name}: {feat} fast={fv} bolt={bv} ({kind}, tol={tol})")
    assert not failures, (
        f"{len(failures)} rigid-class disagreements — "
        "fast and Boltzmann endpoints are diverging on compounds that have no "
        "conformational freedom, which should never happen."
    )


if __name__ == "__main__":
    test_both_endpoints_exist()
    test_reference_compounds_loadable()
    test_both_endpoints_succeed_on_all_compounds()
    test_print_full_comparison_report()
    test_rigid_compounds_agree_tightly()
    print("All 3D flexibility-comparison tests passed!")
