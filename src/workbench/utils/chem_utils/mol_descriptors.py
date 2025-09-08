"""
mol_descriptors.py - Molecular descriptor computation for ADMET modeling

Purpose:
    Computes comprehensive molecular descriptors for ADMET (Absorption, Distribution,
    Metabolism, Excretion, Toxicity) property prediction. Combines RDKit's full
    descriptor set with selected Mordred descriptors and custom stereochemistry features.

Descriptor Categories:
    1. RDKit Descriptors (~200 descriptors)
       - Constitutional (MW, heavy atom count, rotatable bonds)
       - Topological (Balaban J, Kappa indices, Chi indices)
       - Geometric (radius of gyration, spherocity)
       - Electronic (HOMO/LUMO estimates, partial charges)
       - Lipophilicity (LogP, MolLogP)
       - Pharmacophore (H-bond donors/acceptors, aromatic rings)
       - ADMET-specific (TPSA, QED, Lipinski descriptors)

    2. Mordred Descriptors (~110 descriptors from 5 ADMET-relevant modules)
       - AcidBase module: pH-dependent properties (nAcid, nBase)
       - Aromatic module: CYP metabolism features (nAromAtom, nAromBond)
       - Constitutional module: Structural complexity (~40 descriptors including nSpiro, nBridgehead)
       - Chi module: Molecular connectivity indices (~42 descriptors, Chi0-Chi4 variants)
       - CarbonTypes module: Carbon hybridization states for metabolism (~20 descriptors)

    3. Stereochemistry Features (10 custom descriptors)
       - Stereocenter counts (R/S, defined/undefined)
       - Stereobond counts (E/Z, defined/undefined)
       - Stereochemical complexity and coverage metrics
       - Critical for distinguishing drug enantiomers/diastereomers

Pipeline Integration:
    This module expects standardized SMILES from mol_standardize.py:

    1. Standardize structures (mol_standardize.py)
       ↓
    2. Compute descriptors (this module)
       ↓
    3. Feature selection/ML modeling

Output:
    Returns input DataFrame with added descriptor columns:
    - ~220 RDKit descriptors
    - ~85 Mordred descriptors (from 5 modules)
    - 10 stereochemistry descriptors
    Total: ~310 descriptors

    Invalid molecules receive NaN values for all descriptors.

Performance Notes:
    - RDKit descriptors: Fast, vectorized computation
    - Mordred descriptors: Moderate speed
    - Stereochemistry: Moderate speed, requires CIP labeling
    - Memory: <1GB per 10,000 molecules with all descriptors

Special Considerations:
    - Ipc descriptor excluded due to potential overflow issues
    - Molecules failing descriptor calculation get NaN (not dropped)
    - Stereochemistry features optional for non-chiral datasets
    - Salt information from standardization not included in descriptors
      (use separately as categorical feature if needed)
    - Feature selection recommended due to descriptor redundancy

Example Usage:
    import pandas as pd
    from mol_standardize import standardize_dataframe
    from mol_descriptors import compute_descriptors

    # Standard pipeline
    df = pd.read_csv("molecules.csv")
    df = standardize_dataframe(df)  # Standardize first
    df = compute_descriptors(df)    # Then compute descriptors

    # For achiral molecules (faster)
    df = compute_descriptors(df, include_stereochemistry=False)

    # Custom SMILES column
    df = compute_descriptors(df, smiles_column='canonical_smiles')

    # The resulting DataFrame is ready for ML modeling
    X = df.select_dtypes(include=[np.number])  # All numeric descriptors
    y = df['activity']  # Your target variable

References:
    - RDKit descriptors: https://www.rdkit.org/docs/GettingStartedInPython.html#descriptors
    - Mordred: https://github.com/mordred-descriptor/mordred
    - Stereochemistry in drug discovery: https://doi.org/10.1021/acs.jmedchem.0c00915
"""

import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdCIPLabeler
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator as MordredCalculator
from mordred import AcidBase, Aromatic, Constitutional, Chi, CarbonTypes

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def compute_stereochemistry_features(mol):
    """
    Compute stereochemistry descriptors using modern RDKit methods.

    Returns dict with 10 stereochemistry descriptors commonly used in ADMET.
    """
    if mol is None:
        return {
            "num_stereocenters": np.nan,
            "num_unspecified_stereocenters": np.nan,
            "num_defined_stereocenters": np.nan,
            "num_r_centers": np.nan,
            "num_s_centers": np.nan,
            "num_stereobonds": np.nan,
            "num_e_bonds": np.nan,
            "num_z_bonds": np.nan,
            "stereo_complexity": np.nan,
            "frac_defined_stereo": np.nan,
        }

    try:
        # Find all potential stereogenic elements
        stereo_info = Chem.FindPotentialStereo(mol)

        # Initialize counters
        defined_centers = 0
        undefined_centers = 0
        r_centers = 0
        s_centers = 0
        defined_bonds = 0
        undefined_bonds = 0
        e_bonds = 0
        z_bonds = 0

        # Assign CIP labels for accurate R/S and E/Z determination
        rdCIPLabeler.AssignCIPLabels(mol)

        # Process stereogenic elements
        for element in stereo_info:
            if element.type == Chem.StereoType.Atom_Tetrahedral:
                if element.specified == Chem.StereoSpecified.Specified:
                    defined_centers += 1
                    # Get the atom and check its CIP code
                    atom = mol.GetAtomWithIdx(element.centeredOn)
                    if atom.HasProp("_CIPCode"):
                        cip = atom.GetProp("_CIPCode")
                        if cip == "R":
                            r_centers += 1
                        elif cip == "S":
                            s_centers += 1
                else:
                    undefined_centers += 1

            elif element.type == Chem.StereoType.Bond_Double:
                if element.specified == Chem.StereoSpecified.Specified:
                    defined_bonds += 1
                    # Get the bond and check its CIP code
                    bond = mol.GetBondWithIdx(element.centeredOn)
                    if bond.HasProp("_CIPCode"):
                        cip = bond.GetProp("_CIPCode")
                        if cip == "E":
                            e_bonds += 1
                        elif cip == "Z":
                            z_bonds += 1
                else:
                    undefined_bonds += 1

        # Calculate derived metrics
        total_stereocenters = defined_centers + undefined_centers
        total_stereobonds = defined_bonds + undefined_bonds
        total_stereo = total_stereocenters + total_stereobonds

        # Stereochemical complexity (total stereogenic elements)
        stereo_complexity = total_stereo

        # Fraction of defined stereochemistry
        if total_stereo > 0:
            frac_defined = (defined_centers + defined_bonds) / total_stereo
        else:
            frac_defined = 1.0  # No stereo elements = fully defined

        return {
            "num_stereocenters": total_stereocenters,
            "num_unspecified_stereocenters": undefined_centers,
            "num_defined_stereocenters": defined_centers,
            "num_r_centers": r_centers,
            "num_s_centers": s_centers,
            "num_stereobonds": total_stereobonds,
            "num_e_bonds": e_bonds,
            "num_z_bonds": z_bonds,
            "stereo_complexity": stereo_complexity,
            "frac_defined_stereo": frac_defined,
        }

    except Exception as e:
        logger.warning(f"Stereochemistry calculation failed: {e}")
        return {
            "num_stereocenters": np.nan,
            "num_unspecified_stereocenters": np.nan,
            "num_defined_stereocenters": np.nan,
            "num_r_centers": np.nan,
            "num_s_centers": np.nan,
            "num_stereobonds": np.nan,
            "num_e_bonds": np.nan,
            "num_z_bonds": np.nan,
            "stereo_complexity": np.nan,
            "frac_defined_stereo": np.nan,
        }


def compute_descriptors(
    df: pd.DataFrame, smiles_column: str = "smiles", include_stereochemistry: bool = True
) -> pd.DataFrame:
    """
    Compute all molecular descriptors for ADMET modeling.

    Args:
        df: Input DataFrame with SMILES
        smiles_column: Column containing SMILES strings
        include_stereochemistry: Whether to compute stereochemistry features (default True)

    Returns:
        DataFrame with all descriptor columns added

    Example:
        df = standardize(df)  # First standardize
        df = compute_descriptors(df)    # Then compute descriptors with stereo
        df = compute_descriptors(df, include_stereochemistry=False)  # Without stereo
    """
    result = df.copy()

    # Create molecule objects
    logger.info("Creating molecule objects...")
    molecules = []
    for idx, row in result.iterrows():
        smiles = row[smiles_column]

        if pd.isna(smiles) or smiles == "":
            molecules.append(None)
        else:
            mol = Chem.MolFromSmiles(smiles)
            molecules.append(mol)

    # Compute RDKit descriptors
    logger.info("Computing RDKit Descriptors...")

    # Get all RDKit descriptors
    all_descriptors = [x[0] for x in Descriptors._descList]

    # Remove IPC descriptor due to overflow issue
    # See: https://github.com/rdkit/rdkit/issues/1527
    if "Ipc" in all_descriptors:
        all_descriptors.remove("Ipc")

    # Make sure we don't have duplicates
    all_descriptors = list(set(all_descriptors))

    # Initialize calculator
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors)

    # Compute descriptors
    descriptor_values = []
    for mol in molecules:
        if mol is None:
            descriptor_values.append([np.nan] * len(all_descriptors))
        else:
            try:
                values = calc.CalcDescriptors(mol)
                descriptor_values.append(values)
            except Exception as e:
                logger.warning(f"RDKit descriptor calculation failed: {e}")
                descriptor_values.append([np.nan] * len(all_descriptors))

    # Create RDKit features DataFrame
    rdkit_features_df = pd.DataFrame(descriptor_values, columns=calc.GetDescriptorNames(), index=result.index)

    # Add RDKit features to result
    result = pd.concat([result, rdkit_features_df], axis=1)

    # Compute Mordred descriptors
    logger.info("Computing Mordred descriptors from relevant modules...")
    calc = MordredCalculator()

    # Register 5 ADMET-focused modules (avoiding overlap with RDKit)
    calc.register(AcidBase)  # ~2 descriptors: nAcid, nBase
    calc.register(Aromatic)  # ~2 descriptors: nAromAtom, nAromBond
    calc.register(Constitutional)  # ~40 descriptors: structural complexity
    calc.register(Chi)  # ~42 descriptors: connectivity indices
    calc.register(CarbonTypes)  # ~20 descriptors: carbon hybridization

    # Compute Mordred descriptors
    valid_mols = [mol if mol is not None else Chem.MolFromSmiles("C") for mol in molecules]
    mordred_df = calc.pandas(valid_mols, nproc=1)

    # Replace values for invalid molecules with NaN
    for i, mol in enumerate(molecules):
        if mol is None:
            mordred_df.iloc[i] = np.nan

    # Handle Mordred's special error values
    for col in mordred_df.columns:
        mordred_df[col] = pd.to_numeric(mordred_df[col], errors="coerce")

    # Set index to match result DataFrame
    mordred_df.index = result.index

    # Add Mordred features to result
    result = pd.concat([result, mordred_df], axis=1)

    # Compute stereochemistry features if requested
    if include_stereochemistry:
        logger.info("Computing Stereochemistry Descriptors...")

        stereo_features = []
        for mol in molecules:
            stereo_dict = compute_stereochemistry_features(mol)
            stereo_features.append(stereo_dict)

        # Create stereochemistry DataFrame
        stereo_df = pd.DataFrame(stereo_features, index=result.index)

        # Add stereochemistry features to result
        result = pd.concat([result, stereo_df], axis=1)

        logger.info(f"Added {len(stereo_df.columns)} stereochemistry descriptors")

    # Log summary
    valid_mols = sum(1 for m in molecules if m is not None)
    total_descriptors = len(result.columns) - len(df.columns)
    logger.info(f"Computed {total_descriptors} descriptors for {valid_mols}/{len(df)} valid molecules")

    # Log descriptor breakdown
    rdkit_count = len(rdkit_features_df.columns)
    mordred_count = len(mordred_df.columns)
    stereo_count = len(stereo_df.columns) if include_stereochemistry else 0
    logger.info(f"Descriptor breakdown: RDKit={rdkit_count}, Mordred={mordred_count}, Stereo={stereo_count}")
    return result


if __name__ == "__main__":
    # Test the descriptor computation
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 1200)
    print("Testing molecular descriptor computation with stereochemistry")
    print("=" * 60)

    # Create test dataset with stereochemistry
    test_data = pd.DataFrame(
        {
            "smiles": [
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin (no stereo)
                "C[C@H](N)C(=O)O",  # L-Alanine (S, 1 chiral center)
                "C[C@@H](N)C(=O)O",  # D-Alanine (R, 1 chiral center)
                "C[C@@H](O)[C@@H](O)C",  # R,R-2,3-butanediol
                "C[C@@H](O)[C@H](O)C",  # R,S-2,3-butanediol (meso)
                "C/C=C/C=C/C",  # E,E-hexadiene (2 E bonds)
                "C/C=C\\C=C\\C",  # Z,Z-hexadiene (2 Z bonds)
                "C/C=C/C=C\\C",  # E,Z-hexadiene (mixed)
                "CC(C)(C)[C@H](O)[C@@H](O)C(C)(C)C",  # Complex with 2 centers (R,S)
                "CC(F)(Cl)Br",  # Unspecified chiral center
                "C1C[C@H]2CC[C@@H](C1)C2",  # Bicyclic with 2 centers
                "",  # Empty
                "INVALID",  # Invalid
            ],
            "name": [
                "Aspirin",
                "L-Alanine",
                "D-Alanine",
                "R,R-butanediol",
                "meso-butanediol",
                "E,E-hexadiene",
                "Z,Z-hexadiene",
                "E,Z-hexadiene",
                "Complex-RS",
                "Unspecified-chiral",
                "Bicyclic",
                "Empty",
                "Invalid",
            ],
        }
    )

    print("Input data:")
    print(test_data)

    # Test descriptor computation with stereochemistry
    print("\n" + "=" * 60)
    print("Computing descriptors with stereochemistry...")
    result = compute_descriptors(test_data, include_stereochemistry=True)

    # Check total descriptors
    original_cols = len(test_data.columns)
    total_descriptors = len(result.columns) - original_cols

    print(f"\nTotal descriptors computed: {total_descriptors}")

    # Show stereochemistry features for test molecules
    print("\nStereochemistry features:")
    stereo_cols = [
        "num_stereocenters",
        "num_unspecified_stereocenters",
        "num_defined_stereocenters",
        "num_r_centers",
        "num_s_centers",
        "num_stereobonds",
        "num_e_bonds",
        "num_z_bonds",
        "stereo_complexity",
        "frac_defined_stereo",
    ]

    for idx, name in enumerate(test_data["name"]):
        if name not in ["Empty", "Invalid"]:
            print(f"\n{name}:")
            for col in stereo_cols:
                if col in result.columns:
                    val = result.iloc[idx][col]
                    if not pd.isna(val):
                        if col == "frac_defined_stereo":
                            print(f"  {col}: {val:.2f}")
                        else:
                            print(f"  {col}: {val:.0f}")

    # Test without stereochemistry
    print("\n" + "=" * 60)
    print("Computing descriptors WITHOUT stereochemistry...")
    result_no_stereo = compute_descriptors(test_data, include_stereochemistry=False)
    total_descriptors_no_stereo = len(result_no_stereo.columns) - original_cols
    print(f"Total descriptors without stereo: {total_descriptors_no_stereo}")
    print(f"Difference: {total_descriptors - total_descriptors_no_stereo} stereo descriptors")

    """
    Test script for molecular standardization with salt handling
    Demonstrates how salts affect molecular descriptors
    """
    from workbench.utils.chem_utils.mol_standardize import standardize

    # Test with DataFrame including various salt forms
    test_data = pd.DataFrame(
        {
            "smiles": [
                # Acetic acid family (same parent, different salts)
                "CC(=O)O",  # Acetic acid (parent)
                "[Na+].CC(=O)[O-]",  # Sodium acetate
                "[K+].CC(=O)[O-]",  # Potassium acetate
                "[Ca+2].CC(=O)[O-].CC(=O)[O-]",  # Calcium acetate
                "CC(=O)O.CCN",  # Acetic acid + ethylamine salt
                # Tautomer examples
                "Oc1ccccn1",  # 2-hydroxypyridine (tautomer 1)
                "O=c1cccc[nH]1",  # 2-pyridone (tautomer 2)
                "CC(O)=CC(C)=O",  # Acetylacetone enol form
                "CC(=O)CC(C)=O",  # Acetylacetone keto form
                # Isoproterenol family (drug with different salts)
                "CC(C)NCC(O)c1ccc(O)c(O)c1",  # Isoproterenol (free base)
                "CC(C)NCC(O)c1ccc(O)c(O)c1.Cl",  # Isoproterenol HCl
                "CC(C)NCC(O)c1ccc(O)c(O)c1.[Br-]",  # Isoproterenol HBr
                "CC(C)NCC(O)c1ccc(O)c(O)c1.OS(=O)(=O)O",  # Isoproterenol sulfate
                # Carbonic acid family (all should give same parent)
                "[Na+].[Na+].[O-]C([O-])=O",  # Sodium carbonate
                "[K+].[K+].[O-]C([O-])=O",  # Potassium carbonate
                "[Ca+2].[O-]C([O-])=O",  # Calcium carbonate
                # Simple organic (no salt)
                "CC(C)(C)c1ccccc1",  # tert-butylbenzene
            ],
            "compound_id": [f"C{i:03d}" for i in range(1, 18)],
            "logS": [
                4.5,
                5.2,
                5.1,
                4.8,
                4.3,  # Acetic acid family
                7.3,
                7.3,
                6.1,
                6.1,  # Tautomers (should converge)
                3.2,
                3.5,
                3.4,
                2.9,  # Isoproterenol family
                0.05,
                0.95,
                -2.18,  # Carbonates
                5.5,
            ],  # tert-butylbenzene
        }
    )

    print("=" * 80)
    print("MOLECULAR STANDARDIZATION TEST WITH SALT EFFECTS")
    print("=" * 80)

    # Define columns for display
    show_cols = ["compound_id", "smiles", "salt"]
    descriptor_cols = ["MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors", "TPSA", "MolLogP"]

    # ============================================================================
    # TEST 1: Standardization WITH salt extraction (uses parent only)
    # ============================================================================
    print("\n1. STANDARDIZATION WITH SALT EXTRACTION (Parent Only)")
    print("-" * 60)

    test_data_parents = standardize(test_data, extract_salts=True)
    print("\nStandardized molecules (parent forms):")
    print(test_data_parents[["compound_id", "orig_smiles", "smiles", "salt"]])

    # Compute descriptors on parent molecules
    result_parents = compute_descriptors(test_data_parents)
    print("\nDescriptors computed on PARENT molecules:")
    print(result_parents[show_cols + descriptor_cols])

    # Show which compounds converged to same parent
    print("\nCompounds with identical parents:")
    parent_groups = result_parents.groupby("smiles")["compound_id"].apply(list)
    for parent, compounds in parent_groups.items():
        if len(compounds) > 1:
            print(f"  {parent[:30]:30} : {', '.join(compounds)}")

    # ============================================================================
    # TEST 2: Standardization WITHOUT salt extraction (includes salts)
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. STANDARDIZATION WITHOUT SALT EXTRACTION (Including Salts)")
    print("-" * 60)

    test_data_with_salts = standardize(test_data, extract_salts=False, canonicalize_tautomer=True)
    print("\nStandardized molecules (with salts included):")
    print(test_data_with_salts[["compound_id", "orig_smiles", "smiles", "salt"]])

    # Compute descriptors on molecules WITH salts
    result_with_salts = compute_descriptors(test_data_with_salts)
    print("\nDescriptors computed on molecules WITH SALTS:")
    print(result_with_salts[show_cols + descriptor_cols])

    # ============================================================================
    # TEST 3: Compare descriptor differences
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. DESCRIPTOR DIFFERENCES (With Salts - Parent Only)")
    print("-" * 60)

    # Merge results for comparison
    comparison = pd.merge(
        result_parents[["compound_id", "MolWt", "HeavyAtomCount", "TPSA"]],
        result_with_salts[["compound_id", "MolWt", "HeavyAtomCount", "TPSA"]],
        on="compound_id",
        suffixes=("_parent", "_with_salt"),
    )

    # Calculate differences
    comparison["MolWt_diff"] = comparison["MolWt_with_salt"] - comparison["MolWt_parent"]
    comparison["HeavyAtom_diff"] = comparison["HeavyAtomCount_with_salt"] - comparison["HeavyAtomCount_parent"]
    comparison["TPSA_diff"] = comparison["TPSA_with_salt"] - comparison["TPSA_parent"]

    # Show compounds with differences (i.e., those with salts)
    significant_diff = comparison[(comparison["MolWt_diff"].abs() > 0.01) | (comparison["HeavyAtom_diff"] != 0)]

    if not significant_diff.empty:
        print("\nCompounds showing descriptor changes due to salts:")
        print(significant_diff[["compound_id", "MolWt_diff", "HeavyAtom_diff", "TPSA_diff"]])

    # ============================================================================
    # TEST 4: Tautomer convergence check
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. TAUTOMER CONVERGENCE CHECK")
    print("-" * 60)

    # Check tautomer pairs
    tautomer_pairs = [("C006", "C007", "2-hydroxypyridine/2-pyridone"), ("C008", "C009", "Acetylacetone enol/keto")]

    print("\nTautomer standardization (should converge to same canonical form):")
    for id1, id2, name in tautomer_pairs:
        smiles1 = result_parents[result_parents["compound_id"] == id1]["smiles"].values[0]
        smiles2 = result_parents[result_parents["compound_id"] == id2]["smiles"].values[0]
        match = "✓ SAME" if smiles1 == smiles2 else "✗ DIFFERENT"
        print(f"  {name:30} {id1} vs {id2}: {match}")
        if smiles1 == smiles2:
            print(f"    Canonical form: {smiles1}")

    # ============================================================================
    # TEST 5: Salt family analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("5. SALT FAMILY ANALYSIS")
    print("-" * 60)

    families = {
        "Acetic acid": ["C001", "C002", "C003", "C004", "C005"],
        "Isoproterenol": ["C010", "C011", "C012", "C013"],
        "Carbonate": ["C014", "C015", "C016"],
    }

    for family_name, compound_ids in families.items():
        family_data = result_parents[result_parents["compound_id"].isin(compound_ids)]
        unique_parents = family_data["smiles"].nunique()
        parent_smiles = family_data["smiles"].iloc[0]

        print(f"\n{family_name} family:")
        print(f"  Unique parent structures: {unique_parents}")
        print(f"  Parent SMILES: {parent_smiles}")

        # Show salt variations
        salt_data = test_data_parents[test_data_parents["compound_id"].isin(compound_ids)]
        for _, row in salt_data.iterrows():
            salt_info = f" (salt: {row['salt']})" if pd.notna(row["salt"]) else " (no salt)"
            print(f"    {row['compound_id']}: {row['orig_smiles'][:40]:40}{salt_info}")

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)

    print("\n✅ All tests completed!")
