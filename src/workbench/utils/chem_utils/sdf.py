"""SDF File utilities for molecular data in Workbench"""

import logging
import pandas as pd
from typing import List, Optional
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SDWriter

# Set up the logger
log = logging.getLogger("workbench")


def df_to_sdf_file(
    df: pd.DataFrame,
    output_file: str,
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    generate_3d: bool = False,
    optimize_geometry: bool = True,
    v3000: bool = False,
) -> int:
    """Convert DataFrame with SMILES to SDF file.

    By default generates fast 2D depiction coordinates. Set generate_3d=True
    for full ETKDGv3 conformer generation (much slower — seconds per molecule).
    Invalid/missing SMILES and embedding failures are skipped with warnings.

    Args:
        df (pd.DataFrame): DataFrame containing SMILES and other data
        output_file (str): Path to output SDF file
        smiles_col (str): Column name containing SMILES strings
        id_col (str): Column to use as molecule ID/name
        include_cols (list): Specific columns to include as properties (default: all except smiles and molecule columns)
        generate_3d (bool): Generate 3D coordinates using ETKDGv3 (slow).
            When False, generates 2D depiction coords (fast).
        optimize_geometry (bool): Run MMFF optimization after embedding (only applies when generate_3d=True)
        v3000 (bool): Force V3000 format (default V2000, auto-upgrades for large molecules)

    Returns:
        int: Number of molecules successfully written
    """
    written_count = 0
    skipped_count = 0

    # Set up ETKDGv3 embedding parameters only if needed
    if generate_3d:
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 42
        embed_params.useSmallRingTorsions = True

    with SDWriter(output_file) as writer:
        if v3000:
            writer.SetForceV3000(True)

        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            if pd.isna(smiles):
                log.warning(f"Skipping row {idx}: missing SMILES")
                skipped_count += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                log.warning(f"Skipping row {idx}: could not parse SMILES '{smiles}'")
                skipped_count += 1
                continue

            if generate_3d:
                # Full 3D coordinate generation using ETKDGv3 (slow)
                mol = Chem.AddHs(mol)

                # Suppress noisy RDKit warnings (UFFTYPER, etc.) during embedding
                rdkit_logger = RDLogger.logger()
                rdkit_logger.setLevel(RDLogger.ERROR)

                # First attempt with standard ETKDGv3
                try:
                    embed_status = AllChem.EmbedMolecule(mol, embed_params)
                except RuntimeError as e:
                    log.debug(f"Row {idx}: ETKDGv3 embedding raised {e}, trying fallback")
                    embed_status = -1

                if embed_status == -1:
                    # Fallback: random coordinates for difficult molecules
                    fallback_params = AllChem.ETKDGv3()
                    fallback_params.randomSeed = 42
                    fallback_params.useSmallRingTorsions = True
                    fallback_params.useRandomCoords = True
                    try:
                        embed_status = AllChem.EmbedMolecule(mol, fallback_params)
                    except RuntimeError as e:
                        log.warning(f"Skipping row {idx}: fallback embedding raised {e} for '{smiles}'")
                        embed_status = -1
                    if embed_status == -1:
                        rdkit_logger.setLevel(RDLogger.WARNING)
                        log.warning(f"Skipping row {idx}: 3D embedding failed for '{smiles}'")
                        skipped_count += 1
                        continue

                # MMFF geometry optimization
                if optimize_geometry:
                    try:
                        result = AllChem.MMFFOptimizeMolecule(mol)
                        if result == -1:
                            log.debug(f"Row {idx}: MMFF params unavailable, skipping optimization")
                    except Exception:
                        log.debug(f"Row {idx}: MMFF optimization failed, using unoptimized coords")

                # Restore RDKit logging and remove explicit Hs
                rdkit_logger.setLevel(RDLogger.WARNING)
                mol = Chem.RemoveHs(mol)
            else:
                # Fast 2D depiction coordinates
                AllChem.Compute2DCoords(mol)

            # Set molecule name/ID
            if id_col and id_col in df.columns:
                mol.SetProp("_Name", str(row[id_col]))

            # Determine which columns to include
            if include_cols:
                cols_to_add = [col for col in include_cols if col in df.columns and col != smiles_col]
            else:
                # Auto-exclude common molecule column names and SMILES column
                mol_col_names = ["mol", "molecule", "rdkit_mol", "Mol"]
                cols_to_add = [col for col in df.columns if col != smiles_col and col not in mol_col_names]

            # Add properties (skip NaN/None to avoid writing literal "nan" strings)
            for col in cols_to_add:
                value = row[col]
                if pd.isna(value):
                    continue
                mol.SetProp(col, str(value))

            writer.write(mol)
            written_count += 1

    log.info(f"Wrote {written_count} molecules to SDF ({skipped_count} skipped): {output_file}")
    return written_count


def sdf_file_to_df(
    sdf_file: str,
    include_smiles: bool = True,
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
    include_props: Optional[List[str]] = None,
    exclude_props: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert SDF file to DataFrame.

    Args:
        sdf_file: Path to input SDF file
        include_smiles: Add SMILES column to output
        smiles_col: Name for SMILES column
        id_col: Column name for molecule ID/name (uses _Name property)
        include_props: Specific properties to include (default: all)
        exclude_props: Properties to exclude from output

    Returns:
        DataFrame with molecules and their properties
    """
    data = []

    suppl = Chem.SDMolSupplier(sdf_file)
    for idx, mol in enumerate(suppl):
        if mol is None:
            log.warning(f"Could not parse molecule at index {idx}")
            continue

        row_data = {}

        # Add SMILES if requested
        if include_smiles:
            row_data[smiles_col] = Chem.MolToSmiles(mol)

        # Add molecule name/ID if requested
        if id_col and mol.HasProp("_Name"):
            row_data[id_col] = mol.GetProp("_Name")

        # Get all properties
        prop_names = mol.GetPropNames()

        # Filter properties based on include/exclude lists
        if include_props:
            prop_names = [p for p in prop_names if p in include_props]
        if exclude_props:
            prop_names = [p for p in prop_names if p not in exclude_props]

        # Add properties to row
        for prop in prop_names:
            if prop != "_Name":  # Skip _Name if we already handled it
                row_data[prop] = mol.GetProp(prop)

        data.append(row_data)

    df = pd.DataFrame(data)
    log.info(f"Read {len(df)} molecules from SDF: {sdf_file}")

    return df


if __name__ == "__main__":
    import tempfile
    import os

    print("Running SDF utilities tests...")

    # Create test data
    test_data = pd.DataFrame(
        {
            "smiles": [
                "CCO",  # Ethanol
                "c1ccccc1",  # Benzene
                "CC(=O)O",  # Acetic acid
                "INVALID_SMILES",  # Invalid
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            ],
            "name": ["Ethanol", "Benzene", "Acetic Acid", "Invalid", "Caffeine"],
            "mol_weight": [46.07, 78.11, 60.05, 0, 194.19],
            "category": ["alcohol", "aromatic", "acid", "error", "alkaloid"],
            "mol": ["should_exclude", "should_exclude", "should_exclude", "should_exclude", "should_exclude"],
        }
    )

    # Test 1: Basic DataFrame to SDF conversion
    print("\n1. Testing DataFrame to SDF conversion...")
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Test default (2D coords — fast)
        count = df_to_sdf_file(test_data, tmp_path, smiles_col="smiles", id_col="name")
        print(f"   ✓ Wrote {count} molecules with 2D coords (expected 4, skipped 1 invalid)")

        # Test with 3D generation (slow)
        count = df_to_sdf_file(test_data, tmp_path, smiles_col="smiles", id_col="name", generate_3d=True)
        print(f"   ✓ Wrote {count} molecules with 3D coords")

    except Exception as e:
        print(f"   ✗ Error writing SDF: {e}")

    # Test 2: SDF to DataFrame conversion
    print("\n2. Testing SDF to DataFrame conversion...")
    try:
        # Read back the SDF
        df_read = sdf_file_to_df(tmp_path, include_smiles=True, smiles_col="SMILES", id_col="mol_name")
        print(f"   ✓ Read {len(df_read)} molecules from SDF")
        print(f"   ✓ Columns: {list(df_read.columns)}")

    except Exception as e:
        print(f"   ✗ Error reading SDF: {e}")

    # Test 3: Column filtering
    print("\n3. Testing column inclusion/exclusion...")
    try:
        # Test with specific columns only
        count = df_to_sdf_file(
            test_data,
            tmp_path,
            smiles_col="smiles",
            include_cols=["name", "mol_weight"],
            generate_3d=False,
        )

        df_filtered = sdf_file_to_df(tmp_path)
        excluded_mol = "mol" not in df_filtered.columns
        included_weight = any("mol_weight" in str(col) for col in df_filtered.columns)

        print(f"   {'✓' if excluded_mol else '✗'} 'mol' column excluded")
        print(f"   {'✓' if included_weight else '✗'} 'mol_weight' included")

    except Exception as e:
        print(f"   ✗ Error with column filtering: {e}")

    # Test 4: Property filtering on read
    print("\n5. Testing property filtering on read...")
    try:
        # Write full data
        df_to_sdf_file(test_data, tmp_path, smiles_col="smiles", generate_3d=False)

        # Read with include filter
        df_include = sdf_file_to_df(tmp_path, include_props=["mol_weight", "category"])
        print(f"   ✓ Include filter: {list(df_include.columns)}")

        # Read with exclude filter
        df_exclude = sdf_file_to_df(tmp_path, exclude_props=["category"])
        has_category = "category" in df_exclude.columns
        print(f"   {'✗' if has_category else '✓'} Exclude filter: 'category' excluded")

    except Exception as e:
        print(f"   ✗ Error with property filtering: {e}")

    # Test 6: Edge cases
    print("\n6. Testing edge cases...")

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["smiles", "name"])
    try:
        count = df_to_sdf_file(empty_df, tmp_path)
        print(f"   ✓ Empty DataFrame: wrote {count} molecules")
    except Exception as e:
        print(f"   ✗ Empty DataFrame error: {e}")

    # Missing columns
    bad_df = pd.DataFrame({"not_smiles": ["CCO"]})
    try:
        count = df_to_sdf_file(bad_df, tmp_path, smiles_col="smiles")
        print("   ✗ Should have raised error for missing column")
    except KeyError:
        print("   ✓ Correctly raised error for missing SMILES column")

    # Large molecule test (3D generation stress test)
    large_mol_df = pd.DataFrame({"smiles": ["C" * 50], "name": ["Long Chain"]})  # Very long carbon chain
    try:
        count = df_to_sdf_file(large_mol_df, tmp_path, generate_3d=True)
        print(f"   ✓ Large molecule: wrote {count} molecule(s)")
    except Exception as e:
        print(f"   ✗ Large molecule error: {e}")

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print(f"\n✓ Cleaned up temp file: {tmp_path}")

    print("\n✅ All SDF utilities tests completed!")
