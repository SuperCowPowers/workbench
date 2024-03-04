"""Chem/RDKIT utilities for Sageworks"""

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


if __name__ == "__main__":
    # Show a molecule
    show("CC(CN1CC(C)OC(C)C1)Cc2ccc(cc2)C(C)(C)C")
    show("abc")
