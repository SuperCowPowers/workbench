"""Chem/RDKIT utilities for Sageworks"""

import numpy as np
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


def micromolar_to_log(value_uM: float) -> float:
    """
    Convert a concentration in µM (micromolar) to its logarithmic value (log10).

    Parameters:
    value_uM (float): Concentration in micromolar.

    Returns:
    float: Logarithmic value (log10).
    """
    # Replace 0 or negative values with a small number to avoid log errors
    adjusted_value = max(value_uM, 1e-10)

    value_mol_per_l = adjusted_value * 1e-6  # Convert µM to mol/L
    log_value = np.log10(value_mol_per_l)
    return log_value


def log_to_micromolar(log_value):
    """
    Convert a logarithmic value (log10) back to concentration in µM (micromolar).

    Parameters:
    log_value (float): Logarithmic value (log10).

    Returns:
    float: Concentration in micromolar.
    """
    value_mol_per_l = 10**log_value  # Convert log10 back to mol/L
    value_uM = value_mol_per_l * 1e6  # Convert mol/L to µM
    return value_uM


if __name__ == "__main__":
    # Show a molecule
    show("CC(CN1CC(C)OC(C)C1)Cc2ccc(cc2)C(C)(C)C")
    show("abc")

    # Test the concentration conversion functions
    value_uM = 100
    log_value = micromolar_to_log(value_uM)
    print(f"{value_uM} µM = {log_value} log10(M)")
    value_uM_back = log_to_micromolar(log_value)
    print(f"{log_value} log10(M) = {value_uM_back} µM")
