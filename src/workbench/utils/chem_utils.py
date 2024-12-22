"""Chem/RDKIT/Mordred utilities for Workbench"""

import logging
import numpy as np
import pandas as pd

# Third Party Imports
try:
    from rdkit import Chem
    from rdkit.Chem import Mol, Descriptors, rdFingerprintGenerator, Draw
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
    from rdkit import RDLogger

    # Set RDKit logger to only show errors or critical messages
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)

    NO_RDKIT = False
except ImportError:
    print("RDKit Python module not found! pip install rdkit")
    NO_RDKIT = True

try:
    from mordred import Calculator
    from mordred import AcidBase, Aromatic, Polarizability, RotatableBond

    NO_MORDRED = False
except ImportError:
    print("Mordred Python module not found! pip install mordred")
    NO_MORDRED = True

# Set up the logger
log = logging.getLogger("workbench")


def display(smiles: str, width: int = 500, height: int = 500) -> None:
    """
    Displays an image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.
        width (int): Width of the image in pixels. Default is 500.
        height (int): Height of the image in pixels. Default is 500.

    Returns:
    None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(width, height))
        img.show()
    else:
        print(f"Invalid SMILES: {smiles}")


def micromolar_to_log(series_µM: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of concentrations in µM (micromolar) to their logarithmic values (log10).

    Parameters:
    series_uM (pd.Series): Series of concentrations in micromolar.

    Returns:
    pd.Series: Series of logarithmic values (log10).
    """
    # Replace 0 or negative values with a small number to avoid log errors
    adjusted_series = series_µM.clip(lower=1e-9)  # Alignment with another project

    series_mol_per_l = adjusted_series * 1e-6  # Convert µM/L to mol/L
    log_series = np.log10(series_mol_per_l)
    return log_series


def log_to_micromolar(log_series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of logarithmic values (log10) back to concentrations in µM (micromolar).

    Parameters:
    log_series (pd.Series): Series of logarithmic values (log10).

    Returns:
    pd.Series: Series of concentrations in micromolar.
    """
    series_mol_per_l = 10**log_series  # Convert log10 back to mol/L
    series_µM = series_mol_per_l * 1e6  # Convert mol/L to µM
    return series_µM


def log_to_category(log_series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of log values to concentration categories.

    Parameters:
    log_series (pd.Series): Series of logarithmic values (log10).

    Returns:
    pd.Series: Series of concentration categories.
    """
    # Create a solubility classification column
    bins = [-float("inf"), -5, -4, float("inf")]
    labels = ["low", "medium", "high"]
    return pd.cut(log_series, bins=bins, labels=labels)


def compute_molecular_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and add all the Molecular Descriptors

    Args:
        df(pd.DataFrame): The DataFrame to process and generate RDKit/Mordred Descriptors

    Returns:
        pd.DataFrame: The input DataFrame with all the RDKit Descriptors added
    """

    # Check for the smiles column (any capitalization)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Compute/add all the Molecular Descriptors
    log.info("Computing Molecular Descriptors...")

    # Conversion to Molecules
    molecules = [Chem.MolFromSmiles(smile) for smile in df[smiles_column]]

    # Now get all the RDKIT Descriptors
    all_descriptors = [x[0] for x in Descriptors._descList]

    # There's an overflow issue that happens with the IPC descriptor, so we'll remove it
    # See: https://github.com/rdkit/rdkit/issues/1527
    if "Ipc" in all_descriptors:
        all_descriptors.remove("Ipc")

    # Make sure we don't have duplicates
    all_descriptors = list(set(all_descriptors))

    # Super useful Molecular Descriptor Calculator Class
    log.info("Computing RDKit Descriptors...")
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors)
    column_names = calc.GetDescriptorNames()
    descriptor_values = [calc.CalcDescriptors(m) for m in molecules]
    rdkit_features_df = pd.DataFrame(descriptor_values, columns=column_names)

    # Report any NaN or Infinite values and drop those rows
    invalid_rows = rdkit_features_df.isna().any(axis=1) | rdkit_features_df.isin([np.inf, -np.inf]).any(axis=1)
    if invalid_rows.any():
        log.warning(f"Rows with NaN/INF found in the RDKit Descriptors DataFrame: {invalid_rows.sum()}")
        log.warning(f"Invalid rows:\n{rdkit_features_df[invalid_rows]}")

        # Remove the invalid rows
        rdkit_features_df = rdkit_features_df[~invalid_rows]

    # Convert all the columns to numeric
    rdkit_features_df = rdkit_features_df.apply(pd.to_numeric)

    # Now compute Mordred Features
    log.info("Computing Mordred Descriptors...")
    descriptor_choice = [AcidBase, Aromatic, Polarizability, RotatableBond]
    calc = Calculator()
    for des in descriptor_choice:
        calc.register(des)
    mordred_df = calc.pandas(molecules, nproc=1)

    # Report any NaN or Infinite values and drop those rows
    invalid_rows = mordred_df.isna().any(axis=1) | mordred_df.isin([np.inf, -np.inf]).any(axis=1)
    if invalid_rows.any():
        log.warning(f"Rows with NaN/INF found in the Mordred Descriptors DataFrame: {invalid_rows.sum()}")
        log.warning(f"Invalid rows:\n{mordred_df[invalid_rows]}")

        # Remove the invalid rows
        mordred_df = mordred_df[~invalid_rows]

    # Convert all the columns to numeric
    mordred_df = mordred_df.apply(pd.to_numeric)

    # Return the DataFrame with the RDKit and Mordred Descriptors added
    output_df = pd.concat([df, rdkit_features_df, mordred_df], axis=1)

    # Return the DataFrame with the RDKit and Mordred Descriptors added
    return output_df


def compute_morgan_fingerprints(df: pd.DataFrame, radius=2, nBits=2048) -> pd.DataFrame:
    """Compute and add Morgan fingerprints to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        radius (int): Radius for the Morgan fingerprint.
        nBits (int): Number of bits for the fingerprint.

    Returns:
        pd.DataFrame: The input DataFrame with the Morgan fingerprints added as bit strings.
    """

    # Check for the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Convert SMILES to RDKit molecule objects (vectorized)
    molecules = df[smiles_column].apply(Chem.MolFromSmiles)

    # Handle invalid molecules
    invalid_smiles = molecules.isna()
    if invalid_smiles.any():
        log.critical(f"Invalid SMILES strings found at indices: {df.index[invalid_smiles].tolist()}")
        molecules = molecules.dropna()
        df = df.loc[molecules.index].reset_index(drop=True)

    # Create a Morgan fingerprint generator
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)

    # Compute Morgan fingerprints (vectorized)
    fingerprints = molecules.apply(lambda mol: (morgan_generator.GetFingerprint(mol).ToBitString() if mol else None))

    # Add the fingerprints to the DataFrame
    df["morgan_fingerprint"] = fingerprints
    return df


def canonicalize(df: pd.DataFrame, remove_mol_col: bool = True) -> pd.DataFrame:
    """
    Generate RDKit's canonical SMILES for each molecule in the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a column named 'SMILES' (case-insensitive).
        remove_mol_col (bool): Whether to drop the intermediate 'rdkit_molecule' column. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with an additional 'canonical_smiles' column and,
                      optionally, the 'rdkit_molecule' column.
    """
    # Identify the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'SMILES' column")

    # Convert SMILES to RDKit molecules
    df["rdkit_molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # Handle invalid SMILES strings
    invalid_indices = df[df["rdkit_molecule"].isna()].index
    if not invalid_indices.empty:
        log.critical(f"Invalid SMILES strings at indices: {invalid_indices.tolist()}")

    # Vectorized canonicalization
    def mol_to_canonical_smiles(mol):
        return Chem.MolToSmiles(mol) if mol else pd.NA

    df["canonical_smiles"] = df["rdkit_molecule"].apply(mol_to_canonical_smiles)

    # Drop intermediate RDKit molecule column if requested
    if remove_mol_col:
        df.drop(columns=["rdkit_molecule"], inplace=True)

    return df


def custom_tautomer_canonicalization(mol: Mol) -> str:
    """Domain-specific processing of a molecule to select the canonical tautomer.

    This function enumerates all possible tautomers for a given molecule and applies
    custom logic to select the canonical form.

    Args:
        mol (Mol): The RDKit molecule for which the canonical tautomer is to be determined.

    Returns:
        str: The SMILES string of the selected canonical tautomer.
    """
    tautomer_enumerator = TautomerEnumerator()
    enumerated_tautomers = tautomer_enumerator.Enumerate(mol)
    for taut in enumerated_tautomers:
        # Custom logic to select the canonical tautomer can go here
        print(Chem.MolToSmiles(taut))

    # Example: return the first tautomer (replace with actual custom selection logic)
    return Chem.MolToSmiles(enumerated_tautomers[0])


def standard_tautomer_canonicalization(mol: Mol) -> str:
    """Standard processing of a molecule to select the canonical tautomer.

    RDKit's `TautomerEnumerator` uses heuristics to select a canonical tautomer,
    such as preferring keto over enol forms and minimizing formal charges.

    Args:
        mol (Mol): The RDKit molecule for which the canonical tautomer is to be determined.

    Returns:
        str: The SMILES string of the canonical tautomer.
    """
    tautomer_enumerator = TautomerEnumerator()
    canonical_tautomer = tautomer_enumerator.Canonicalize(mol)
    return Chem.MolToSmiles(canonical_tautomer)


def perform_tautomerization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform tautomer enumeration and canonicalization on a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.

    Returns:
        pd.DataFrame: A new DataFrame with additional 'canonical_smiles' and 'tautomeric_form' columns.
    """
    # Standardize SMILES strings and create 'rdkit_molecule' column for further processing
    df = canonicalize(df, remove_mol_col=False)

    # Helper function to safely canonicalize a molecule's tautomer
    def safe_tautomerize(mol):
        """Safely canonicalize a molecule's tautomer, handling errors gracefully."""
        if not mol:
            return pd.NA
        try:
            # Use RDKit's standard Tautomer enumeration and canonicalization
            # For custom logic, replace with custom_tautomer_canonicalization(mol)
            return standard_tautomer_canonicalization(mol)
        except Exception as e:
            log.warning(f"Tautomerization failed: {str(e)}")
            return pd.NA

    # Apply tautomer canonicalization to each molecule
    df["tautomeric_form"] = df["rdkit_molecule"].apply(safe_tautomerize)

    # Drop intermediate RDKit molecule column to clean up the DataFrame
    df.drop(columns=["rdkit_molecule"], inplace=True)

    return df


if __name__ == "__main__":

    # Pyridone molecule
    smiles = "C1=CC=NC(=O)C=C1"
    display(smiles)

    # Test the concentration conversion functions
    df = pd.DataFrame({"smiles": [smiles, smiles, smiles, smiles, smiles, smiles], "µM": [500, 50, 5, 1, 0.1, 0]})

    # Convert µM to log10
    df["log10"] = micromolar_to_log(df["µM"])
    print(df)

    # Convert log10 back to µM
    df["µM_new"] = log_to_micromolar(df["log10"])
    print(df)

    # Convert log10 to categories
    df["category"] = log_to_category(df["log10"])
    print(df)

    # Compute Molecular Descriptors
    df = pd.DataFrame({"smiles": [smiles, smiles, smiles, smiles, smiles]})
    df = compute_molecular_descriptors(df)
    print(df)

    # Compute Morgan Fingerprints
    df = compute_morgan_fingerprints(df)
    print(df)

    # Perform Tautomerization
    df = perform_tautomerization(df)
    print(df)
