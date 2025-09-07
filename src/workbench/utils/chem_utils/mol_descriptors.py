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

    2. Mordred Descriptors (4 selected modules)
       - AcidBase: pKa-related features, acidic/basic group counts
       - Aromatic: aromatic ring systems, pi-electron properties
       - Polarizability: molecular polarizability estimates
       - RotatableBond: flexibility measures beyond simple counts

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
    - ~200 RDKit descriptors (prefix: various)
    - ~20 Mordred descriptors (prefix: various)
    - 10 stereochemistry descriptors (prefix: num_, stereo_, frac_)

    Invalid molecules receive NaN values for all descriptors.

Performance Notes:
    - RDKit descriptors: Fast, vectorized computation
    - Mordred descriptors: Slower, can use multiprocessing (nproc parameter)
    - Stereochemistry: Moderate speed, requires CIP labeling
    - Memory: ~1GB per 10,000 molecules with all descriptors

Special Considerations:
    - Ipc descriptor excluded due to potential overflow issues
    - Molecules failing descriptor calculation get NaN (not dropped)
    - Stereochemistry features optional for non-chiral datasets
    - Salt information from standardization not included in descriptors
      (use separately as categorical feature if needed)

Example Usage:
    import pandas as pd
    from mol_standardize import standardize
    from mol_descriptors import compute_descriptors

    # Standard pipeline
    df = pd.read_csv("molecules.csv")
    df = standardize(df)          # Standardize first
    df = compute_descriptors(df)  # Then compute descriptors

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
from mordred import AcidBase, Aromatic, Polarizability, RotatableBond

logger = logging.getLogger(__name__)


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
    logger.info("Computing Mordred Descriptors...")

    # Initialize Mordred with specific descriptor sets for ADMET
    descriptor_choice = [AcidBase, Aromatic, Polarizability, RotatableBond]
    calc = MordredCalculator()
    for des in descriptor_choice:
        calc.register(des)

    # Compute Mordred descriptors
    # Filter out None molecules for Mordred computation
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

    return result


if __name__ == "__main__":
    # Test the descriptor computation
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

    print("\n✅ All tests completed!")
