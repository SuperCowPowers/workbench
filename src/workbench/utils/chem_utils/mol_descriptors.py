"""
mol_descriptors.py - Molecular descriptor computation for ADMET modeling

Purpose:
    Computes comprehensive molecular descriptors for ADMET (Absorption, Distribution,
    Metabolism, Excretion, Toxicity) property prediction. Combines RDKit's full
    descriptor set with selected Mordred descriptors and custom stereochemistry features.

Descriptor Categories:
    1. RDKit Descriptors (~220 descriptors)
       - Constitutional (MW, heavy atom count, rotatable bonds)
       - Topological (Balaban J, Kappa indices, Chi indices)
       - Geometric (radius of gyration, spherocity)
       - Electronic (HOMO/LUMO estimates, partial charges)
       - Lipophilicity (LogP, MolLogP)
       - Pharmacophore (H-bond donors/acceptors, aromatic rings)
       - ADMET-specific (TPSA, QED, Lipinski descriptors)

    2. Mordred Descriptors (~80 descriptors from 5 ADMET-relevant modules)
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
    df = compute_descriptors(df, include_stereo=False)

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
import re
import time
from contextlib import contextmanager
from rdkit import Chem
from rdkit.Chem import Descriptors, rdCIPLabeler
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator as MordredCalculator
from mordred import AcidBase, Aromatic, Constitutional, Chi, CarbonTypes


logger = logging.getLogger("workbench")
logger.setLevel(logging.DEBUG)


# Helper context manager for timing
@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.2f}s")


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


def compute_descriptors(df: pd.DataFrame, include_mordred: bool = True, include_stereo: bool = True) -> pd.DataFrame:
    """
    Compute all molecular descriptors for ADMET modeling.

    Args:
        df: Input DataFrame with SMILES
        include_mordred: Whether to compute Mordred descriptors (default True)
        include_stereo: Whether to compute stereochemistry features (default True)

    Returns:
        DataFrame with all descriptor columns added

    Example:
        df = standardize(df)  # First standardize
        df = compute_descriptors(df)    # Then compute descriptors with stereo
        df = compute_descriptors(df, include_stereo=False)  # Without stereo
        df = compute_descriptors(df, include_mordred=False)  # RDKit only
    """

    # Check for the smiles column (any capitalization)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

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
    rdkit_features_df = pd.DataFrame(descriptor_values, columns=calc.GetDescriptorNames())

    # Add RDKit features to result
    # Remove any columns from result that exist in rdkit_features_df
    result = result.drop(columns=result.columns.intersection(rdkit_features_df.columns))
    result = pd.concat([result, rdkit_features_df], axis=1)

    # Compute Mordred descriptors
    if include_mordred:
        logger.info("Computing Mordred descriptors from relevant modules...")
        calc = MordredCalculator()

        # Register 5 ADMET-focused modules (avoiding overlap with RDKit)
        calc.register(AcidBase)  # ~2 descriptors: nAcid, nBase
        calc.register(Aromatic)  # ~2 descriptors: nAromAtom, nAromBond
        calc.register(Constitutional)  # ~30 descriptors: structural complexity
        calc.register(Chi)  # ~32 descriptors: connectivity indices
        calc.register(CarbonTypes)  # ~20 descriptors: carbon hybridization

        # Compute Mordred descriptors
        valid_mols = [mol if mol is not None else Chem.MolFromSmiles("C") for mol in molecules]
        mordred_df = calc.pandas(valid_mols, nproc=1)  # Endpoint multiprocessing will fail with nproc>1

        # Replace values for invalid molecules with NaN
        for i, mol in enumerate(molecules):
            if mol is None:
                mordred_df.iloc[i] = np.nan

        # Handle Mordred's special error values
        for col in mordred_df.columns:
            mordred_df[col] = pd.to_numeric(mordred_df[col], errors="coerce")

        # Add Mordred features to result
        # Remove any columns from result that exist in mordred
        result = result.drop(columns=result.columns.intersection(mordred_df.columns))
        result = pd.concat([result, mordred_df], axis=1)

    # Compute stereochemistry features if requested
    if include_stereo:
        logger.info("Computing Stereochemistry Descriptors...")

        stereo_features = []
        for mol in molecules:
            stereo_dict = compute_stereochemistry_features(mol)
            stereo_features.append(stereo_dict)

        # Create stereochemistry DataFrame
        stereo_df = pd.DataFrame(stereo_features)

        # Add stereochemistry features to result
        result = result.drop(columns=result.columns.intersection(stereo_df.columns))
        result = pd.concat([result, stereo_df], axis=1)

        logger.info(f"Added {len(stereo_df.columns)} stereochemistry descriptors")

    # Log summary
    valid_mols = sum(1 for m in molecules if m is not None)
    total_descriptors = len(result.columns) - len(df.columns)
    logger.info(f"Computed {total_descriptors} descriptors for {valid_mols}/{len(df)} valid molecules")

    # Log descriptor breakdown
    rdkit_count = len(rdkit_features_df.columns)
    mordred_count = len(mordred_df.columns) if include_mordred else 0
    stereo_count = len(stereo_df.columns) if include_stereo else 0
    logger.info(f"Descriptor breakdown: RDKit={rdkit_count}, Mordred={mordred_count}, Stereo={stereo_count}")

    # Sanitize column names for AWS Athena compatibility
    # - Must be lowercase, no special characters except underscore, no spaces
    result.columns = [re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", col.lower())) for col in result.columns]

    # Drop duplicate columns if any exist after sanitization
    if result.columns.duplicated().any():
        logger.warning("Duplicate column names after sanitization - dropping duplicates!")
        result = result.loc[:, ~result.columns.duplicated()]

    return result


if __name__ == "__main__":
    from mol_standardize import standardize
    from workbench.api import DataSource

    # Configure pandas display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 1200)

    # Test data - stereochemistry examples
    stereo_test_data = pd.DataFrame(
        {
            "smiles": [
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "C[C@H](N)C(=O)O",  # L-Alanine
                "C[C@@H](N)C(=O)O",  # D-Alanine
                "C/C=C/C=C/C",  # E,E-hexadiene
                "CC(F)(Cl)Br",  # Unspecified chiral
                "",
                "INVALID",  # Invalid cases
            ],
            "name": ["Aspirin", "L-Alanine", "D-Alanine", "E,E-hexadiene", "Unspecified", "Empty", "Invalid"],
        }
    )

    # Test data - salt handling examples
    salt_test_data = pd.DataFrame(
        {
            "smiles": [
                "CC(=O)O",  # Acetic acid
                "[Na+].CC(=O)[O-]",  # Sodium acetate
                "CC(C)NCC(O)c1ccc(O)c(O)c1.Cl",  # Drug HCl salt
                "Oc1ccccn1",  # Tautomer 1
                "O=c1cccc[nH]1",  # Tautomer 2
            ],
            "compound_id": [f"C{i:03d}" for i in range(1, 6)],
        }
    )

    def run_basic_tests():
        """Run basic functionality tests"""
        print("=" * 80)
        print("BASIC FUNCTIONALITY TESTS")
        print("=" * 80)

        # Test stereochemistry
        result = compute_descriptors(stereo_test_data, include_stereo=True)

        print("\nStereochemistry features (selected molecules):")
        for idx, name in enumerate(stereo_test_data["name"][:4]):
            print(
                f"{name:15} - centers: {result.iloc[idx]['num_stereocenters']:.0f}, "
                f"R/S: {result.iloc[idx]['num_r_centers']:.0f}/"
                f"{result.iloc[idx]['num_s_centers']:.0f}"
            )

        # Test salt handling
        print("\nSalt extraction test:")
        std_result = standardize(salt_test_data, extract_salts=True)
        for _, row in std_result.iterrows():
            salt_info = f" → salt: {row['salt']}" if pd.notna(row["salt"]) else ""
            print(f"{row['compound_id']}: {row['smiles'][:30]}{salt_info}")

    def run_performance_tests():
        """Run performance timing tests"""
        print("\n" + "=" * 80)
        print("PERFORMANCE TESTS on real world molecules")
        print("=" * 80)

        # Get a real dataset from Workbench
        ds = DataSource("aqsol_data")
        df = ds.pull_dataframe()[["id", "smiles"]][:1000]  # Limit to 1000 for testing
        n_mols = df.shape[0]
        print(f"Pulled {n_mols} molecules from DataSource 'aqsol_data'")

        # Test configurations
        configs = [
            ("Standardize (full)", standardize, {"extract_salts": True, "canonicalize_tautomer": True}),
            ("Standardize (minimal)", standardize, {"extract_salts": False, "canonicalize_tautomer": False}),
            ("Descriptors (all)", compute_descriptors, {"include_mordred": True, "include_stereo": True}),
            ("Descriptors (RDKit only)", compute_descriptors, {"include_mordred": False, "include_stereo": False}),
        ]

        results = []
        for name, func, params in configs:
            start = time.time()
            _ = func(df, **params)
            elapsed = time.time() - start
            throughput = n_mols / elapsed
            results.append((name, elapsed, throughput))
            print(f"{name:25} {elapsed:6.2f}s ({throughput:6.1f} mol/s)")

        # Full pipeline test
        print("\nFull pipeline (standardize + all descriptors):")
        start = time.time()
        std_data = standardize(df)
        standardize_time = time.time() - start
        print(f"  Standardize: {standardize_time:.2f}s ({n_mols / standardize_time:.1f} mol/s)")
        start = time.time()
        _ = compute_descriptors(std_data)
        descriptor_time = time.time() - start
        print(f"  Descriptors: {descriptor_time:.2f}s ({n_mols / descriptor_time:.1f} mol/s)")
        pipeline_time = standardize_time + descriptor_time
        print(f"  Total: {pipeline_time:.2f}s ({n_mols / pipeline_time:.1f} mol/s)")

        return results

    # Run tests
    run_basic_tests()
    timing_results = run_performance_tests()

    print("\n✅ All tests completed!")
