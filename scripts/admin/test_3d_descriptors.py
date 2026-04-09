"""Diagnostic script for 3D descriptor failures on the full compound set.

Pulls the all_molecules dataset (~35k rows) and runs the 3D feature
computation locally to identify and categorize failure modes.

Usage:
    python scripts/admin/test_3d_descriptors.py
"""

import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Workbench Imports
from workbench.api import DFStore
from workbench.utils.chem_utils.mol_standardize import standardize
from workbench.utils.chem_utils.mol_descriptors_3d import (
    compute_descriptors_3d,
    generate_conformers,
    is_too_complex,
    get_3d_feature_names,
    MAX_HEAVY_ATOMS,
    MAX_ROTATABLE_BONDS,
    MAX_RING_SYSTEMS,
)


def diagnose_molecule(smiles: str) -> dict:
    """Diagnose why a molecule might fail 3D descriptor computation.

    Returns a dict with diagnostic info about the molecule.
    """
    info = {"smiles": smiles, "failure_reason": None}

    # Step 1: Can RDKit parse it?
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        info["failure_reason"] = "invalid_smiles"
        return info

    # Step 2: Molecular properties
    info["heavy_atoms"] = mol.GetNumHeavyAtoms()
    info["rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    info["rings"] = mol.GetRingInfo().NumRings()
    info["mol_wt"] = rdMolDescriptors.CalcExactMolWt(mol)

    # Step 3: Complexity check
    if is_too_complex(mol):
        info["failure_reason"] = "too_complex"
        if info["heavy_atoms"] > MAX_HEAVY_ATOMS:
            info["failure_reason"] += f"_heavy_atoms({info['heavy_atoms']})"
        if info["rotatable_bonds"] > MAX_ROTATABLE_BONDS:
            info["failure_reason"] += f"_rot_bonds({info['rotatable_bonds']})"
        if info["rings"] > MAX_RING_SYSTEMS:
            info["failure_reason"] += f"_rings({info['rings']})"
        return info

    # Step 4: Can we generate conformers?
    start = time.time()
    try:
        mol_h = Chem.AddHs(mol)
        conf_mol = generate_conformers(mol_h, n_conformers=10, optimize=True)
        elapsed = time.time() - start
        info["conformer_time_s"] = round(elapsed, 2)

        if conf_mol is None or conf_mol.GetNumConformers() == 0:
            info["failure_reason"] = "conformer_generation_failed"
            return info

        info["n_conformers"] = conf_mol.GetNumConformers()

        # Flag slow molecules (> 5s is concerning for endpoint with 60s timeout)
        if elapsed > 5.0:
            info["failure_reason"] = f"slow_conformer({elapsed:.1f}s)"

    except Exception as e:
        info["conformer_time_s"] = round(time.time() - start, 2)
        info["failure_reason"] = f"conformer_exception: {str(e)[:100]}"

    return info


def run_diagnostics(df: pd.DataFrame, smiles_col: str = "smiles"):
    """Run diagnostics on a DataFrame of molecules."""

    print(f"\n{'=' * 80}")
    print(f"3D Descriptor Diagnostics — {len(df)} molecules")
    print(f"{'=' * 80}")

    # Step 1: Standardize (same as endpoint does)
    print("\n[1/3] Standardizing molecules...")
    df_std = standardize(df, extract_salts=True)

    # Step 2: Run full 3D descriptor computation and track results
    print("[2/3] Computing 3D descriptors (this may take a while)...")
    start = time.time()
    result = compute_descriptors_3d(df_std, n_conformers=10, optimize=True)
    total_time = time.time() - start

    feature_names = get_3d_feature_names()
    first_feature = feature_names[0]

    success_mask = result[first_feature].notna()
    n_success = success_mask.sum()
    n_fail = len(result) - n_success
    fail_pct = 100.0 * n_fail / len(result)

    print(f"\n{'─' * 60}")
    print(f"Overall Results: {n_success}/{len(result)} succeeded, {n_fail} failed ({fail_pct:.1f}%)")
    print(f"Total time: {total_time:.1f}s ({len(result) / total_time:.1f} mol/s)")
    print(f"{'─' * 60}")

    # Step 3: Diagnose failures
    failed_df = result[~success_mask].copy()
    if len(failed_df) == 0:
        print("\nNo failures to diagnose!")
        return result

    print(f"\n[3/3] Diagnosing {len(failed_df)} failures...")
    diagnostics = []
    for idx, row in failed_df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or smiles == "":
            diagnostics.append({"smiles": smiles, "failure_reason": "empty_smiles"})
            continue
        diag = diagnose_molecule(smiles)
        diagnostics.append(diag)

    diag_df = pd.DataFrame(diagnostics)

    # Summarize failure reasons
    print(f"\n{'─' * 60}")
    print("Failure Breakdown:")
    print(f"{'─' * 60}")
    reason_counts = diag_df["failure_reason"].value_counts()
    for reason, count in reason_counts.items():
        print(f"  {reason:45s} {count:5d}  ({100.0 * count / len(result):5.1f}%)")

    # Show property distributions of failures
    if "heavy_atoms" in diag_df.columns:
        numeric_diag = diag_df.dropna(subset=["heavy_atoms"])
        if len(numeric_diag) > 0:
            print(f"\n{'─' * 60}")
            print("Property Distributions of Failed Molecules:")
            print(f"{'─' * 60}")
            for prop in ["heavy_atoms", "rotatable_bonds", "rings", "mol_wt"]:
                if prop in numeric_diag.columns:
                    vals = numeric_diag[prop].dropna()
                    if len(vals) > 0:
                        print(
                            f"  {prop:20s}  min={vals.min():.0f}  median={vals.median():.0f}  "
                            f"max={vals.max():.0f}  mean={vals.mean():.1f}"
                        )

    # Show slowest molecules (by conformer generation time)
    if "conformer_time_s" in diag_df.columns:
        slow = diag_df.dropna(subset=["conformer_time_s"]).nlargest(10, "conformer_time_s")
        if len(slow) > 0:
            print(f"\n{'─' * 60}")
            print("Slowest Failed Molecules (top 10):")
            print(f"{'─' * 60}")
            for _, row in slow.iterrows():
                smiles_short = str(row["smiles"])[:60]
                t = row["conformer_time_s"]
                ha = row.get("heavy_atoms", "?")
                reason = row.get("failure_reason", "unknown")
                print(f"  {t:6.1f}s  atoms={ha}  {reason:30s}  {smiles_short}")

    # Show some example SMILES for each failure category
    print(f"\n{'─' * 60}")
    print("Example SMILES per Failure Category:")
    print(f"{'─' * 60}")
    for reason in reason_counts.index[:10]:
        examples = diag_df[diag_df["failure_reason"] == reason]["smiles"].head(3).tolist()
        print(f"\n  [{reason}]")
        for s in examples:
            print(f"    {str(s)[:100]}")

    return result


def run_corner_case_tests():
    """Test specific molecules that exercise known corner cases."""

    print(f"\n{'=' * 80}")
    print("Corner Case Tests — Known Tricky Molecules")
    print(f"{'=' * 80}")

    test_cases = pd.DataFrame(
        {
            "smiles": [
                # Should pass: simple drug-like molecules
                "c1ccccc1",  # Benzene — flat aromatic
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin — standard drug
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine — rigid
                "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone — steroid
                # Corner case: bridged polycyclic (known embedding difficulty)
                "C1CC2CC1CC2O",  # Norbornanol — bridged bicyclic, public
                # Corner case: boron (unsupported by UFF/MMFF, caused C++ crash)
                "OB(O)c1ccccc1",  # Phenylboronic acid
                "CC1=CC(=CC(=C1)B(O)O)C",  # 3,5-dimethylphenylboronic acid
                # Corner case: complex drug with deuterium labels
                "[2H]C([2H])(F)Oc1ccc(cc1)C(=O)NC",  # Deuterated drug fragment
                # Corner case: large flexible molecule
                "CCCCCCCCCCCCCCCCCCCC",  # Eicosane — long chain, very flexible
                # Corner case: metal-containing / unusual
                "[Na+].CC(=O)[O-]",  # Sodium acetate — salt
                # Corner case: boron-containing compounds (UFF/MMFF edge cases)
                "OB(O)c1ccc(C(F)(F)F)cc1",                      # 4-(trifluoromethyl)phenylboronic acid
                "O=Cc1ccc(B(O)O)cc1",                            # 4-formylphenylboronic acid
                "F[B-](F)(F)F.[K+]",                             # Potassium tetrafluoroborate — ionic boron
                # Corner case: highly constrained polycyclic (optimizer crash)
                "C12C3C4C1C5C3C2C45",                            # Cubane — dense cage, public
                # TEMP: row 266 crash investigation — delete after confirming fix
                r"C/C=C(\C)C(=O)O[C@H]1C[C@@H](OC(C)=O)[C@@]2(C(=O)OC)CO[C@H]3[C@@H](O)[C@@](C)([C@]45O[C@@]4(C)[C@H]4C[C@@H]5O[C@@H]5OC=C[C@@]54O)[C@H]4[C@]1(CO[C@]4(O)C(=O)OC)[C@@H]32",
            ],
            "label": [
                "benzene",
                "aspirin",
                "caffeine",
                "testosterone",
                "bridged_bicyclic",
                "phenylboronic_acid",
                "dimethylphenylboronic_acid",
                "deuterated_fragment",
                "eicosane_flexible",
                "sodium_acetate_salt",
                "trifluoromethyl_boronic_acid",
                "formyl_boronic_acid",
                "potassium_tetrafluoroborate",
                "cubane_bridged",
                "TEMP_row266_crash",
            ],
        }
    )

    n_pass = 0
    n_fail = 0

    for _, row in test_cases.iterrows():
        smiles = row["smiles"]
        label = row["label"]

        # Run standardization + 3D descriptors on single molecule
        single_df = pd.DataFrame({"smiles": [smiles]})
        try:
            std_df = standardize(single_df, extract_salts=True)
            result = compute_descriptors_3d(std_df, n_conformers=10, optimize=True)
            feature_names = get_3d_feature_names()
            has_values = result[feature_names[0]].notna().iloc[0]

            if has_values:
                status = "PASS"
                n_pass += 1
                # Show a few key descriptors
                pmi1 = result["pmi1"].iloc[0]
                asph = result["asphericity"].iloc[0]
                detail = f"pmi1={pmi1:.1f}  asphericity={asph:.3f}"
            else:
                status = "SKIP"
                n_fail += 1
                detail = "NaN (conformer generation or descriptors failed)"

        except Exception as e:
            status = "FAIL"
            n_fail += 1
            detail = f"Exception: {str(e)[:80]}"

        emoji = {"PASS": "+", "SKIP": "-", "FAIL": "!"}[status]
        print(f"  [{emoji}] {status:4s}  {label:30s}  {detail}")

    print(f"\n  Results: {n_pass} passed, {n_fail} failed/skipped out of {len(test_cases)}")
    return n_fail == 0


if __name__ == "__main__":
    # First: run corner case tests
    all_passed = run_corner_case_tests()

    if not all_passed:
        print("\nCorner case tests had failures — review before running full dataset")

    # Then: run full dataset diagnostics
    df_store = DFStore()
    print("\nPulling all_molecules dataset...")
    df = df_store.get("/harmony/datasets/all_molecules_df_20260325")
    print(f"Dataset shape: {df.shape}")

    # Run diagnostics (use sample_size=None for full dataset, or set a number for testing)
    result = run_diagnostics(df[270:280])
