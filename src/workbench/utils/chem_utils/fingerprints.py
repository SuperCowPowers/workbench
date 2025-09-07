"""Molecular fingerprint computation utilities"""

import logging
import pandas as pd

# Molecular Descriptor Imports
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

# Set up the logger
log = logging.getLogger("workbench")


def compute_morgan_fingerprints(df: pd.DataFrame, radius=2, n_bits=2048, counts=True) -> pd.DataFrame:
    """Compute and add Morgan fingerprints to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        radius (int): Radius for the Morgan fingerprint.
        n_bits (int): Number of bits for the fingerprint.
        counts (bool): Count simulation for the fingerprint.

    Returns:
        pd.DataFrame: The input DataFrame with the Morgan fingerprints added as bit strings.

    Note:
        See: https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html
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

    # Convert SMILES to RDKit molecule objects (vectorized)
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        delete_mol_column = True
        df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)
        # Make sure our molecules are not None
        failed_smiles = df[df["molecule"].isnull()][smiles_column].tolist()
        if failed_smiles:
            log.error(f"Failed to convert the following SMILES to molecules: {failed_smiles}")
        df = df.dropna(subset=["molecule"])

    # If we have fragments in our compounds, get the largest fragment before computing fingerprints
    largest_frags = df["molecule"].apply(
        lambda mol: rdMolStandardize.LargestFragmentChooser().choose(mol) if mol else None
    )

    # Create a Morgan fingerprint generator
    if counts:
        n_bits *= 4  # Multiply by 4 to simulate counts
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, countSimulation=counts)

    # Compute Morgan fingerprints (vectorized)
    fingerprints = largest_frags.apply(
        lambda mol: (morgan_generator.GetFingerprint(mol).ToBitString() if mol else pd.NA)
    )

    # Add the fingerprints to the DataFrame
    df["fingerprint"] = fingerprints

    # Drop the intermediate 'molecule' column if it was added
    if delete_mol_column:
        del df["molecule"]
    return df


if __name__ == "__main__":
    print("Running molecular fingerprint tests...")
    print("Note: This requires molecular_screening module to be available")

    # Test molecules
    test_molecules = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",  # With stereochemistry
        "sodium_acetate": "CC(=O)[O-].[Na+]",  # Salt
        "benzene": "c1ccccc1",
        "butene_e": "C/C=C/C",  # E-butene
        "butene_z": "C/C=C\\C",  # Z-butene
    }

    # Test 1: Morgan Fingerprints
    print("\n1. Testing Morgan fingerprint generation...")

    test_df = pd.DataFrame({"SMILES": list(test_molecules.values()), "name": list(test_molecules.keys())})

    fp_df = compute_morgan_fingerprints(test_df.copy(), radius=2, n_bits=512, counts=False)

    print("   Fingerprint generation results:")
    for _, row in fp_df.iterrows():
        fp = row.get("fingerprint", "N/A")
        fp_len = len(fp) if fp != "N/A" else 0
        print(f"   {row['name']:15} → {fp_len} bits")

    # Test 2: Different fingerprint parameters
    print("\n2. Testing different fingerprint parameters...")

    # Test with counts enabled
    fp_counts_df = compute_morgan_fingerprints(test_df.copy(), radius=3, n_bits=256, counts=True)

    print("   With count simulation (256 bits * 4):")
    for _, row in fp_counts_df.iterrows():
        fp = row.get("fingerprint", "N/A")
        fp_len = len(fp) if fp != "N/A" else 0
        print(f"   {row['name']:15} → {fp_len} bits")

    # Test 3: Edge cases
    print("\n3. Testing edge cases...")

    # Invalid SMILES
    invalid_df = pd.DataFrame({"SMILES": ["INVALID", ""]})
    try:
        fp_invalid = compute_morgan_fingerprints(invalid_df.copy())
        print(f"   ✓ Invalid SMILES handled: {len(fp_invalid)} valid molecules")
    except Exception as e:
        print(f"   ✓ Invalid SMILES properly raised error: {type(e).__name__}")

    # Test with pre-existing molecule column
    mol_df = test_df.copy()
    mol_df["molecule"] = mol_df["SMILES"].apply(Chem.MolFromSmiles)
    fp_with_mol = compute_morgan_fingerprints(mol_df)
    print(f"   ✓ Pre-existing molecule column handled: {len(fp_with_mol)} fingerprints generated")

    print("\n✅ All fingerprint tests completed!")
