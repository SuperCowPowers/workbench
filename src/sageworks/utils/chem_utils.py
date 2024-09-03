"""Chem/RDKIT utilities for Sageworks"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def show(smiles: str, size: tuple[int, int] = (500, 500)) -> None:
    """
    Displays an image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.
        size (tuple[int, int]): A tuple specifying width and height of the image.

    Returns:
    None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=size)
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
    adjusted_series = series_µM.clip(lower=1e-10)

    series_mol_per_l = adjusted_series * 1e-6  # Convert µM to mol/L
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


if __name__ == "__main__":

    # Show a molecule
    smiles = "CC(CN1CC(C)OC(C)C1)Cc2ccc(cc2)C(C)(C)C"
    show(smiles)

    # Test the concentration conversion functions
    df = pd.DataFrame({"smiles": [smiles, smiles, smiles, smiles, smiles], "µM": [100, 10, 1, 0.1, 0.01]})

    # Convert µM to log10
    df["log10"] = micromolar_to_log(df["µM"])
    print(df)

    # Convert log10 back to µM
    df["µM_new"] = log_to_micromolar(df["log10"])
    print(df)
