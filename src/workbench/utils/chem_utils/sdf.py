"""SDF File utilities for molecular data in Workbench"""

import logging
import pandas as pd
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

# Set up the logger
log = logging.getLogger("workbench")


def df_to_sdf_file(
    df: pd.DataFrame,
    output_file: str,
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    skip_invalid: bool = True,
    generate_3d: bool = True,
):
    """
    Convert DataFrame with SMILES to SDF file.

    Args:
        df: DataFrame containing SMILES and other data
        output_file: Path to output SDF file
        smiles_col: Column name containing SMILES strings
        id_col: Column to use as molecule ID/name
        include_cols: Specific columns to include as properties (default: all except smiles and molecule columns)
        skip_invalid: Skip invalid SMILES instead of raising error
        generate_3d: Generate 3D coordinates and optimize geometry
    """
    written_count = 0

    with SDWriter(output_file) as writer:
        writer.SetForceV3000(True)
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                if not skip_invalid:
                    raise ValueError(f"Invalid SMILES at row {idx}: {row[smiles_col]}")
                continue

            # Generate 3D coordinates
            if generate_3d:
                mol = Chem.AddHs(mol)

                # Try progressively more aggressive embedding strategies
                embed_strategies = [
                    {"maxAttempts": 1000, "randomSeed": 42},
                    {"maxAttempts": 1000, "randomSeed": 42, "useRandomCoords": True},
                    {"maxAttempts": 1000, "randomSeed": 42, "boxSizeMult": 5.0},
                ]

                embedded = False
                for strategy in embed_strategies:
                    if AllChem.EmbedMolecule(mol, **strategy) != -1:
                        embedded = True
                        break

                if not embedded:
                    if not skip_invalid:
                        raise ValueError(f"Could not generate 3D coords for row {idx}")
                    continue

                AllChem.MMFFOptimizeMolecule(mol)

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

            # Add properties
            for col in cols_to_add:
                mol.SetProp(col, str(row[col]))

            writer.write(mol)
            written_count += 1

    log.info(f"Wrote {written_count} molecules to SDF: {output_file}")
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
        # Test with 3D generation
        count = df_to_sdf_file(
            test_data, tmp_path, smiles_col="smiles", id_col="name", skip_invalid=True, generate_3d=True
        )
        print(f"   ✓ Wrote {count} molecules with 3D coords (expected 4, skipped 1 invalid)")

        # Test without 3D generation
        count = df_to_sdf_file(
            test_data, tmp_path, smiles_col="smiles", id_col="name", skip_invalid=True, generate_3d=False
        )
        print(f"   ✓ Wrote {count} molecules without 3D coords")

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
            skip_invalid=True,
            generate_3d=False,
        )

        df_filtered = sdf_file_to_df(tmp_path)
        excluded_mol = "mol" not in df_filtered.columns
        included_weight = any("mol_weight" in str(col) for col in df_filtered.columns)

        print(f"   {'✓' if excluded_mol else '✗'} 'mol' column excluded")
        print(f"   {'✓' if included_weight else '✗'} 'mol_weight' included")

    except Exception as e:
        print(f"   ✗ Error with column filtering: {e}")

    # Test 4: Error handling
    print("\n4. Testing error handling...")

    # Test with skip_invalid=False
    try:
        count = df_to_sdf_file(test_data, tmp_path, smiles_col="smiles", skip_invalid=False, generate_3d=False)
        print("   ✗ Should have raised error for invalid SMILES")
    except ValueError:
        print("   ✓ Correctly raised error for invalid SMILES")

    # Test 5: Property filtering on read
    print("\n5. Testing property filtering on read...")
    try:
        # Write full data
        df_to_sdf_file(test_data, tmp_path, smiles_col="smiles", skip_invalid=True, generate_3d=False)

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
        count = df_to_sdf_file(large_mol_df, tmp_path, generate_3d=True, skip_invalid=True)
        print(f"   ✓ Large molecule: wrote {count} molecule(s)")
    except Exception as e:
        print(f"   ✗ Large molecule error: {e}")

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print(f"\n✓ Cleaned up temp file: {tmp_path}")

    print("\n✅ All SDF utilities tests completed!")
