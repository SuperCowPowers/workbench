"""
mol_tagging.py - Molecular property tagging for ADMET modeling
Adds a 'tags' column to DataFrames for filtering and classification
"""

import logging
from typing import List, Set, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol, Descriptors

logger = logging.getLogger(__name__)


# ============================================================================
# Property Detection Functions (Internal)
# ============================================================================


def _get_metal_tags(mol: Mol) -> Set[str]:
    """Detect metal-related tags."""
    tags = set()
    if mol is None:
        return tags

    # Metalloenzyme-relevant metals
    metalloenzyme_metals = {"Zn", "Cu", "Fe", "Mn", "Co", "Ni", "Mo", "V"}

    # Heavy/toxic metals
    heavy_metals = {"Pb", "Hg", "Cd", "As", "Cr", "Tl", "Ba", "Be", "Al", "Sb", "Se", "Bi", "Ag"}

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in metalloenzyme_metals:
            tags.add("metalloenzyme_metal")
        if symbol in heavy_metals:
            tags.add("heavy_metal")

    return tags


def _get_halogen_tags(mol: Mol) -> Set[str]:
    """Detect halogenation patterns."""
    tags = set()
    if mol is None:
        return tags

    # Count halogens
    halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ["F", "Cl", "Br", "I"])

    if halogen_count > 0:
        tags.add("halogenated")

    # Flag heavily halogenated compounds
    heavy_atom_count = mol.GetNumHeavyAtoms()
    if heavy_atom_count > 0:
        halogen_ratio = halogen_count / heavy_atom_count
        if halogen_ratio > 0.5 or halogen_count > 4:
            tags.add("highly_halogenated")

    return tags


def _get_druglike_tags(mol: Mol) -> Set[str]:
    """Assess drug-likeness properties."""
    tags = set()
    if mol is None:
        return tags

    # Calculate descriptors once
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    # Lipinski's Rule of Five
    ro5_violations = 0
    if mw > 500:
        ro5_violations += 1
    if logp > 5:
        ro5_violations += 1
    if hbd > 5:
        ro5_violations += 1
    if hba > 10:
        ro5_violations += 1

    if ro5_violations <= 1:
        tags.add("ro5_pass")
    if ro5_violations == 0:
        tags.add("ro5_strict")

    # Veber's rules
    if rotatable <= 10 and tpsa <= 140:
        tags.add("veber_pass")

    # Lead-like
    if 150 <= mw <= 350 and -3 <= logp <= 3.5:
        tags.add("lead_like")

    # Fragment-like (Rule of Three)
    if mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3:
        tags.add("fragment_like")

    # Size categories
    if mw < 200:
        tags.add("small_molecule")
    elif mw > 700:
        tags.add("large_molecule")

    return tags


def _get_structural_tags(mol: Mol) -> Set[str]:
    """Detect structural features."""
    tags = set()
    if mol is None:
        return tags

    # Check for multiple fragments
    if len(Chem.GetMolFrags(mol)) > 1:
        tags.add("multi_fragment")

    # Check for rings
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        tags.add("acyclic")
    else:
        tags.add("cyclic")
        # Check for aromatic rings by checking if any ring atoms are aromatic
        for ring in ring_info.AtomRings():
            if any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                tags.add("aromatic")
                break

    # Check for chirality
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        tags.add("chiral")

    return tags


# ============================================================================
# Main Tagging Function
# ============================================================================


def tag_molecules(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    tag_column: str = "tags",
    tag_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add molecular property tags to a DataFrame.

    Designed to work after mol_standardize.py processing.
    Adds a single 'tags' column containing a list of string tags.

    Args:
        df: Input DataFrame with SMILES
        smiles_column: Column containing SMILES strings
        tag_column: Name for output tags column (default: "tags")
        tag_categories: Which tag categories to include. Options:
            - "metals": Metal content tags
            - "halogens": Halogenation tags
            - "druglike": Drug-likeness assessments
            - "structure": Structural features
            - None (default): Include all categories

    Returns:
        DataFrame with tags column added

    Example:
        df = tag_molecules(df)  # Add all tags
        df = tag_molecules(df, tag_categories=["druglike"])  # Only drug-likeness
    """
    result = df.copy()

    # Default to all categories
    if tag_categories is None:
        tag_categories = ["metals", "halogens", "druglike", "structure"]

    # Initialize tags column
    all_tags = []

    # Process each molecule
    for idx, row in result.iterrows():
        # Parse SMILES to molecule
        smiles = row[smiles_column]
        if pd.isna(smiles) or smiles == "":
            all_tags.append(["invalid_smiles"])
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            all_tags.append(["invalid_smiles"])
            continue

        # Collect tags based on categories
        tags = set()

        if "metals" in tag_categories:
            tags.update(_get_metal_tags(mol))

        if "halogens" in tag_categories:
            tags.update(_get_halogen_tags(mol))

        if "druglike" in tag_categories:
            tags.update(_get_druglike_tags(mol))

        if "structure" in tag_categories:
            tags.update(_get_structural_tags(mol))

        # Convert to sorted list for consistency
        all_tags.append(sorted(list(tags)))

    # Add tags column
    result[tag_column] = all_tags

    # Log summary
    total = len(result)
    valid = sum(1 for tags in all_tags if "invalid_smiles" not in tags)
    ro5_pass = sum(1 for tags in all_tags if "ro5_pass" in tags)

    logger.info(f"Tagged {total} molecules: {valid} valid, {ro5_pass} pass Ro5")

    return result


# ============================================================================
# Utility Functions
# ============================================================================


def filter_by_tags(
    df: pd.DataFrame, require: Optional[List[str]] = None, exclude: Optional[List[str]] = None, tag_column: str = "tags"
) -> pd.DataFrame:
    """
    Filter DataFrame rows based on tags.

    Args:
        df: DataFrame with tags column
        require: Tags that must be present (AND logic)
        exclude: Tags that must not be present
        tag_column: Name of tags column

    Returns:
        Filtered DataFrame

    Example:
        # Get drug-like molecules without heavy metals
        filtered = filter_by_tags(df,
                                 require=["ro5_pass"],
                                 exclude=["heavy_metal"])
    """
    result = df.copy()

    if require:
        for tag in require:
            result = result[result[tag_column].apply(lambda x: tag in x)]

    if exclude:
        for tag in exclude:
            result = result[result[tag_column].apply(lambda x: tag not in x)]

    logger.info(f"Filtered {len(df)} → {len(result)} molecules")

    return result


def get_tag_summary(df: pd.DataFrame, tag_column: str = "tags") -> pd.Series:
    """
    Get summary statistics of tags in DataFrame.

    Args:
        df: DataFrame with tags column
        tag_column: Name of tags column

    Returns:
        Series with tag counts
    """
    # Flatten all tags and count
    all_tags = []
    for tags_list in df[tag_column]:
        all_tags.extend(tags_list)

    tag_counts = pd.Series(all_tags).value_counts()
    return tag_counts


if __name__ == "__main__":
    # Test the tagging functionality
    print("Testing molecular tagging system")
    print("=" * 60)

    # Create test dataset
    test_data = pd.DataFrame(
        {
            "smiles": [
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                "C" * 50,  # Large alkane
                "C(Cl)(Cl)(Cl)Cl",  # Carbon tetrachloride
                "[Zn+2].[Cl-].[Cl-]",  # Zinc chloride
                "CCC",  # Propane
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
                "[Pb+2].[O-]C(=O)C",  # Lead acetate
                "",  # Empty
                "INVALID_SMILES",  # Invalid
            ],
            "compound_id": [f"C{i:03d}" for i in range(1, 11)],
        }
    )

    print("Input data:")
    print(test_data[["compound_id", "smiles"]])

    # Apply tagging
    print("\n" + "=" * 60)
    print("Applying molecular tags...")
    tagged_df = tag_molecules(test_data)

    print("\nTagged results:")
    for _, row in tagged_df.iterrows():
        tags_str = ", ".join(row["tags"]) if row["tags"] else "none"
        print(f"{row['compound_id']}: {tags_str}")

    # Test filtering
    print("\n" + "=" * 60)
    print("Testing filters...")

    # Get drug-like molecules
    druglike = filter_by_tags(tagged_df, require=["ro5_pass"])
    print(f"Drug-like molecules: {list(druglike['compound_id'])}")

    # Exclude problematic molecules
    clean = filter_by_tags(tagged_df, exclude=["heavy_metal", "highly_halogenated", "invalid_smiles"])
    print(f"Clean molecules: {list(clean['compound_id'])}")

    # Get tag summary
    print("\n" + "=" * 60)
    print("Tag summary:")
    summary = get_tag_summary(tagged_df)
    print(summary.head(10))

    print("\n✅ All tests completed!")
