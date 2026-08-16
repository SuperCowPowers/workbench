"""Molecular fingerprint computation utilities for ADMET modeling.

This module provides Morgan count fingerprints, the standard for ADMET prediction.
Count fingerprints outperform binary fingerprints for molecular property prediction.

References:
    - Count vs Binary: https://pubs.acs.org/doi/10.1021/acs.est.3c02198
    - ECFP/Morgan: https://pubs.acs.org/doi/10.1021/ci100050t
"""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Suppress RDKit warnings (e.g., "not removing hydrogen atom without neighbors")
# Keep errors enabled so we see actual problems
RDLogger.DisableLog("rdApp.warning")

# Set up the logger
log = logging.getLogger("workbench")


def similarity_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    counts: bool = True,
    fp_size: int | None = None,
) -> tuple[list, list[int]]:
    """Morgan fingerprints as RDKit objects, for Tanimoto similarity work.

    The counterpart to `feature_fingerprints`: that one folds to a fixed-width
    numeric column because models and projections need one, while similarity has no
    such constraint and is computed straight off the fingerprint objects. Both default
    to counts, whose Tanimoto is the multiset (MinMax) form.

    Args:
        smiles_list: SMILES strings
        radius: Morgan radius (default 2 = ECFP4 equivalent)
        counts: Count fingerprints (default) or binary
        fp_size: Fold into this many bits. None (default) keeps the fingerprint sparse,
            so unrelated substructures never collide -- folding only exists to satisfy
            fixed-width consumers.

    Returns:
        Tuple of (fingerprints, positions) where positions gives each fingerprint's index
        in smiles_list. Unparseable entries are omitted from both, so the two are shorter
        than the input whenever any SMILES fails.

    Example:
        >>> from rdkit import DataStructs
        >>> fps, positions = similarity_fingerprints(df["smiles"].tolist())
        >>> sims = DataStructs.BulkTanimotoSimilarity(fps[0], fps)
    """
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    fp_gen = GetMorganGenerator(radius=radius, fpSize=fp_size or 2048)
    if counts:
        build = fp_gen.GetCountFingerprint if fp_size else fp_gen.GetSparseCountFingerprint
    else:
        build = fp_gen.GetFingerprint if fp_size else fp_gen.GetSparseFingerprint

    fps, positions = [], []
    for i, smiles in enumerate(smiles_list):
        # Guard non-strings (e.g. NaN) before RDKit — it raises TypeError on floats
        if not isinstance(smiles, str) or not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(build(mol))
            positions.append(i)
    return fps, positions


def feature_fingerprints(df: pd.DataFrame, radius: int = 2, n_bits: int = 4096) -> pd.DataFrame:
    """Compute Morgan count fingerprints for ADMET modeling.

    Generates true count fingerprints where each bit position contains the
    number of times that substructure appears in the molecule (clamped to 0-255).
    This is the recommended approach for ADMET prediction per 2025 research.

    Args:
        df: Input DataFrame containing SMILES strings.
        radius: Radius for the Morgan fingerprint (default 2 = ECFP4 equivalent).
        n_bits: Number of bits for the fingerprint (default 4096). Wide enough to
            limit count-corrupting bit collisions (a collision sums two unrelated
            substructure counts).

    Returns:
        pd.DataFrame: Input DataFrame with 'fingerprint' column added.
                      Values are comma-separated uint8 counts.

    Note:
        Count fingerprints outperform binary for ADMET prediction.
        See: https://pubs.acs.org/doi/10.1021/acs.est.3c02198

    Future Me:
        This silently drops rows whose SMILES RDKit can't parse, so the output
        has fewer rows than the input. Callers downstream (FingerprintProximity,
        residual_features, UQModelV1) currently work around this by pre-validating
        SMILES at their own layer — which means MolFromSmiles runs twice per
        SMILES on the inference path. The cleaner fix is one of:
          (a) preserve the input's DataFrame index when dropping rows so
              callers can derive a survivor mask via `df_out.index`, or
          (b) keep all rows and put NaN in the fingerprint column for failures
              (more pandas/sklearn-idiomatic).
        Either would let `FingerprintProximity.neighbors_from_query_df` drop
        its redundant pre-validate. Deferred — see the workaround in
        fingerprint_proximity.py.
    """
    delete_mol_column = False

    # Check for the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Sanity check the molecule column (sometimes it gets serialized, which doesn't work)
    if "molecule" in df.columns and df["molecule"].dtype == "string":
        log.warning("Detected serialized molecules in 'molecule' column. Removing...")
        del df["molecule"]

    # Convert SMILES to RDKit molecule objects
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        delete_mol_column = True

        def _safe_mol_from_smiles(smi):
            if smi is None or pd.isna(smi) or not isinstance(smi, str) or not smi:
                return None
            return Chem.MolFromSmiles(smi)

        df["molecule"] = df[smiles_column].apply(_safe_mol_from_smiles)
        # Make sure our molecules are not None
        failed_smiles = df[df["molecule"].isnull()][smiles_column].tolist()
        if failed_smiles:
            log.warning(f"Failed to convert {len(failed_smiles)} SMILES to molecules ({failed_smiles})")
        df = df.dropna(subset=["molecule"]).copy()

    # If we have fragments in our compounds, get the largest fragment before computing fingerprints
    largest_frags = df["molecule"].apply(
        lambda mol: rdMolStandardize.LargestFragmentChooser().choose(mol) if mol else None
    )

    def mol_to_count_string(mol):
        """Convert molecule to comma-separated count fingerprint string."""
        if mol is None:
            return pd.NA

        # Get hashed Morgan fingerprint with counts
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)

        # Initialize array and populate with counts (clamped to uint8 range)
        counts = np.zeros(n_bits, dtype=np.uint8)
        for idx, count in fp.GetNonzeroElements().items():
            counts[idx] = min(count, 255)

        # Return as comma-separated string
        return ",".join(map(str, counts))

    # Compute Morgan count fingerprints
    fingerprints = largest_frags.apply(mol_to_count_string)

    # Add the fingerprints to the DataFrame
    df["fingerprint"] = fingerprints

    # Drop the intermediate 'molecule' column if it was added
    if delete_mol_column:
        del df["molecule"]

    return df


if __name__ == "__main__":
    print("Running Morgan count fingerprint tests...")

    # Test molecules
    test_molecules = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",  # With stereochemistry
        "sodium_acetate": "CC(=O)[O-].[Na+]",  # Salt (largest fragment used)
        "benzene": "c1ccccc1",
        "butene_e": "C/C=C/C",  # E-butene
        "butene_z": "C/C=C\\C",  # Z-butene
    }

    # Test 1: Morgan Count Fingerprints (default parameters)
    print("\n1. Testing Morgan fingerprint generation (radius=2, n_bits=4096)...")

    test_df = pd.DataFrame({"SMILES": list(test_molecules.values()), "name": list(test_molecules.keys())})
    fp_df = feature_fingerprints(test_df.copy())

    print("   Fingerprint generation results:")
    for _, row in fp_df.iterrows():
        fp = row.get("fingerprint", "N/A")
        if pd.notna(fp):
            counts = [int(x) for x in fp.split(",")]
            non_zero = sum(1 for c in counts if c > 0)
            max_count = max(counts)
            print(f"   {row['name']:15} → {len(counts)} features, {non_zero} non-zero, max={max_count}")
        else:
            print(f"   {row['name']:15} → N/A")

    # Test 2: Different parameters
    print("\n2. Testing with different parameters (radius=3, n_bits=1024)...")

    fp_df_custom = feature_fingerprints(test_df.copy(), radius=3, n_bits=1024)

    for _, row in fp_df_custom.iterrows():
        fp = row.get("fingerprint", "N/A")
        if pd.notna(fp):
            counts = [int(x) for x in fp.split(",")]
            non_zero = sum(1 for c in counts if c > 0)
            print(f"   {row['name']:15} → {len(counts)} features, {non_zero} non-zero")
        else:
            print(f"   {row['name']:15} → N/A")

    # Test 3: Edge cases
    print("\n3. Testing edge cases...")

    # Invalid SMILES
    invalid_df = pd.DataFrame({"SMILES": ["INVALID", ""]})
    fp_invalid = feature_fingerprints(invalid_df.copy())
    print(f"   ✓ Invalid SMILES handled: {len(fp_invalid)} rows returned")

    # Test with pre-existing molecule column
    mol_df = test_df.copy()
    mol_df["molecule"] = mol_df["SMILES"].apply(Chem.MolFromSmiles)
    fp_with_mol = feature_fingerprints(mol_df)
    print(f"   ✓ Pre-existing molecule column handled: {len(fp_with_mol)} fingerprints generated")

    # Test 4: Verify count values are reasonable
    print("\n4. Verifying count distribution...")
    all_counts = []
    for _, row in fp_df.iterrows():
        fp = row.get("fingerprint", "N/A")
        if pd.notna(fp):
            counts = [int(x) for x in fp.split(",")]
            all_counts.extend([c for c in counts if c > 0])

    if all_counts:
        print(f"   Non-zero counts: min={min(all_counts)}, max={max(all_counts)}, mean={np.mean(all_counts):.2f}")

    print("\n✅  All fingerprint tests completed!")
