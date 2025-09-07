"""
mol_descriptors.py - Molecular descriptor computation for ADMET modeling
Computes all RDKit and Mordred descriptors with exact feature name compatibility
"""

import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator as MordredCalculator
from mordred import AcidBase, Aromatic, Polarizability, RotatableBond

logger = logging.getLogger(__name__)


def compute_descriptors(df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
    """
    Compute all molecular descriptors for ADMET modeling.

    Args:
        df: Input DataFrame with SMILES
        smiles_column: Column containing SMILES strings

    Returns:
        DataFrame with all descriptor columns added

    Example:
        df = standardize_dataframe(df)  # First standardize
        df = compute_descriptors(df)    # Then compute descriptors
    """
    result = df.copy()

    # Create molecule objects
    logger.info("Creating molecule objects...")
    molecules = []
    for idx, row in result.iterrows():
        smiles = row[smiles_column]

        if pd.isna(smiles) or smiles == '':
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
    valid_mols = [mol if mol is not None else Chem.MolFromSmiles('C') for mol in molecules]

    mordred_df = calc.pandas(valid_mols, nproc=1)

    # Replace values for invalid molecules with NaN
    for i, mol in enumerate(molecules):
        if mol is None:
            mordred_df.iloc[i] = np.nan

    # Handle Mordred's special error values
    for col in mordred_df.columns:
        mordred_df[col] = pd.to_numeric(mordred_df[col], errors='coerce')

    # Set index to match result DataFrame
    mordred_df.index = result.index

    # Add Mordred features to result
    result = pd.concat([result, mordred_df], axis=1)

    # Log summary
    valid_mols = sum(1 for m in molecules if m is not None)
    total_descriptors = len(result.columns) - len(df.columns)
    logger.info(f"Computed {total_descriptors} descriptors for {valid_mols}/{len(df)} valid molecules")

    return result


if __name__ == "__main__":
    # Test the descriptor computation
    print("Testing molecular descriptor computation")
    print("=" * 60)

    # Create test dataset
    test_data = pd.DataFrame({
        'smiles': [
            'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            '',  # Empty
            'INVALID',  # Invalid
        ],
        'name': ['Aspirin', 'Caffeine', 'Ibuprofen', 'Empty', 'Invalid']
    })

    print("Input data:")
    print(test_data)

    # Test descriptor computation
    print("\n" + "=" * 60)
    print("Computing descriptors...")
    result = compute_descriptors(test_data)

    # Check total descriptors
    original_cols = len(test_data.columns)
    total_descriptors = len(result.columns) - original_cols

    print(f"\nTotal descriptors computed: {total_descriptors}")

    # Show sample values for Aspirin
    print("\nAspirin descriptor values (sample):")
    aspirin_row = result.iloc[0]
    print(f"  MolWt: {aspirin_row.get('MolWt', 'N/A'):.2f}")
    print(f"  MolLogP: {aspirin_row.get('MolLogP', 'N/A'):.2f}")
    print(f"  TPSA: {aspirin_row.get('TPSA', 'N/A'):.2f}")
    print(f"  nAcid: {aspirin_row.get('nAcid', 'N/A')}")
    print(f"  nAromatic: {aspirin_row.get('nAromatic', 'N/A')}")

    print("\n✅ All tests completed!")