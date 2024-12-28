"""Chem/RDKIT/Mordred utilities for Workbench"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
import base64

# Workbench Imports
from workbench.utils.pandas_utils import feature_quality_metrics

# Molecular Descriptor Imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Mol, Descriptors, rdFingerprintGenerator, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
    from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcExactMolWt
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


def img_from_smiles(smiles: str, width: int = 500, height: int = 500, dark_mode: bool = True) -> Optional[str]:
    """
    Generate an image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.
        width (int): Width of the image in pixels. Default is 500.
        height (int): Height of the image in pixels. Default is 500.
        dark_mode (bool): Use dark mode for the image. Default is True.

    Returns:
        str: Base64-encoded image of the molecule or None if the SMILES string is invalid.
    """

    # Set up the drawing options
    dos = Draw.MolDrawOptions()
    if dark_mode:
        rdMolDraw2D.SetDarkMode(dos)
    dos.setBackgroundColour((0, 0, 0, 0))

    # Convert the SMILES string to an RDKit molecule and generate the image
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, options=dos, size=(width, height))
        return img
    else:
        log.warning(f"Invalid SMILES: {smiles}")
        return None


def show(smiles: str, width: int = 500, height: int = 500) -> None:
    """
    Displays an image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.
        width (int): Width of the image in pixels. Default is 500.
        height (int): Height of the image in pixels. Default is 500.

    Returns:
    None
    """
    img = img_from_smiles(smiles, width, height)
    if img:
        img.show()


def svg_from_smiles(smiles: str) -> Optional[str]:
    """
    Generate an SVG image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.

    Returns:
        str: SVG image of the molecule or None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # mol = Chem.AddHs(mol)

        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Initialize the SVG drawer with desired dimensions
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)

        # Draw the molecule
        drawer.DrawMolecule(mol)

        # Finalize the drawing
        drawer.FinishDrawing()

        # Retrieve the SVG string
        svg = drawer.GetDrawingText()

        # Clean the SVG string
        svg_cleaned = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
        svg_cleaned = svg_cleaned.replace("xmlns:rdkit='http://www.rdkit.org/xml'", "")

        # Encode the SVG string
        # Note: html.Img(
        #           src=f"data:image/svg+xml;base64,{encoded_svg}",
        #        ),
        encoded_svg = base64.b64encode(svg.encode("utf-8")).decode()
        return encoded_svg
    else:
        return None


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


def remove_disconnected_fragments(mol: Mol) -> Optional[Mol]:
    """
    Remove disconnected fragments from a molecule, keeping the largest fragment with at least one heavy atom.

    Args:
        mol (Mol): RDKit molecule object.

    Returns:
        Optional[Mol]: The largest fragment with a heavy atom, or None if no such fragment exists.
    """
    # Get all fragments as individual molecules
    fragments = Chem.GetMolFrags(mol, asMols=True)

    # Filter fragments with at least one heavy atom
    fragments_with_heavy_atoms = [frag for frag in fragments if frag.GetNumHeavyAtoms() > 0]

    # Return the largest fragment with heavy atoms, or None if no valid fragments
    if fragments_with_heavy_atoms:
        return max(fragments_with_heavy_atoms, key=lambda frag: frag.GetNumAtoms())
    else:
        return None


def contains_heavy_metals(mol):
    """
    Check if a molecule contains any heavy metals (broad filter).

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if any heavy metals are detected, False otherwise.
    """
    heavy_metals = {"Zn", "Cu", "Fe", "Mn", "Co", "Pb", "Hg", "Cd", "As"}
    return any(atom.GetSymbol() in heavy_metals for atom in mol.GetAtoms())


def contains_toxic_elements(mol):
    """
    Check if a molecule contains toxic elements.

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if toxic elements are detected, False otherwise.
    """
    # Elements that are toxic in most contexts
    always_toxic = {"Pb", "Hg", "Cd", "As", "Be", "Tl", "Sb"}

    # Elements that are context-dependent for toxicity
    conditional_toxic = {"Se", "Cr"}
    for atom in mol.GetAtoms():
        element = atom.GetSymbol()
        if element in always_toxic:
            return True
        if element in conditional_toxic:
            # Context-dependent checks (e.g., oxidation state or bonding)
            if element == "Cr" and atom.GetFormalCharge() == 6:  # Chromium (VI)
                return True
            if element == "Se" and atom.GetFormalCharge() != 0:  # Selenium (non-neutral forms)
                return True

    # No toxic elements detected
    return False


def contains_toxic_groups(mol):
    """
    Check if a molecule contains known toxic groups.

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if toxic groups are detected, False otherwise.
    """
    toxic_smarts = [
        "[Cr](=O)(=O)=O",  # Chromium (VI)
        "[As](=O)(=O)-[OH]",  # Arsenic oxide
        "[Hg]",  # Mercury atom
        "[Pb]",  # Lead atom
        "[Se][Se]",  # Diselenide compounds
        "[Be]",  # Beryllium atom
        "[Tl+]",  # Thallium ion
    ]
    for pattern in toxic_smarts:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return True
    return False


def contains_metalloenzyme_relevant_metals(mol):
    """
    Check if a molecule contains metals relevant to metalloenzymes.

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if metalloenzyme-relevant metals are detected, False otherwise.
    """
    metalloenzyme_metals = {"Zn", "Cu", "Fe", "Mn", "Co"}
    return any(atom.GetSymbol() in metalloenzyme_metals for atom in mol.GetAtoms())


def contains_salts(mol):
    """
    Check if a molecule contains common salts or counterions.

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if salts are detected, False otherwise.
    """
    # Define common inorganic salt fragments (SMARTS patterns)
    salt_patterns = ["[Na+]", "[K+]", "[Cl-]", "[Mg+2]", "[Ca+2]", "[NH4+]", "[SO4--]"]
    for pattern in salt_patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return True
    return False


def is_druglike_compound(mol: Mol) -> bool:
    """
    Filter for drug-likeness and QSAR relevance based on Lipinski's Rule of Five.
    Returns False for molecules unlikely to be orally bioavailable.

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if the molecule is drug-like, False otherwise.
    """

    # Lipinski's Rule of Five
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    if mw > 500 or logp > 5 or hbd > 5 or hba > 10:
        return False

    # Allow exceptions for linear molecules that meet strict RO5 criteria
    if mol.GetRingInfo().NumRings() == 0:
        if mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3:
            pass  # Allow small, non-cyclic druglike compounds
        else:
            return False

    return True


def add_compound_tags(df, mol_column="mol"):
    """
    Adds a 'compound_tags' column to a DataFrame, tagging molecules based on their properties.

    Args:
        df (pd.DataFrame): Input DataFrame containing molecular data.
        mol_column (str): Column name containing RDKit molecule objects.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'compound_tags' column.
    """
    # Initialize the compound_tags column
    df["compound_tags"] = [[] for _ in range(len(df))]

    # Process each molecule in the DataFrame
    for idx, row in df.iterrows():
        mol = row[mol_column]
        tags = []

        # Check for salts
        if contains_salts(mol):
            tags.append("salt")

        # Check for fragments (should be done after salt check)
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if len(fragments) > 1:
            tags.append("frag")
            mol = remove_disconnected_fragments(mol)  # Keep largest fragment
            if mol is None:  # Handle largest fragment with no heavy atoms
                tags.append("no_heavy_atoms")
                log.warning(f"No heavy atoms: {row['smiles']}")
                df.at[idx, "compound_tags"] = tags
                continue

        # Check for heavy metals
        if contains_heavy_metals(mol):
            tags.append("heavy_metals")

        # Check for toxic elements
        if contains_toxic_elements(mol):
            tags.append("toxic_element")
        if contains_toxic_groups(mol):
            tags.append("toxic_group")

        # Check for metalloenzyme-relevant metals
        if contains_metalloenzyme_relevant_metals(mol):
            tags.append("metalloenzyme")

        # Check for drug-likeness
        if is_druglike_compound(mol):
            tags.append("druglike")

        # Update tags and molecule
        df.at[idx, "compound_tags"] = tags
        df.at[idx, mol_column] = mol  # Update molecule after processing (if needed)

    return df


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
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        df["molecule"] = [Chem.MolFromSmiles(smile) for smile in df[smiles_column]]

    # Add Compound Tags
    df = add_compound_tags(df, mol_column="molecule")

    # Now get all the RDKIT Descriptors
    all_descriptors = [x[0] for x in Descriptors._descList]

    # There's an overflow issue that happens with the IPC descriptor, so we'll remove it
    # See: https://github.com/rdkit/rdkit/issues/1527
    if "Ipc" in all_descriptors:
        all_descriptors.remove("Ipc")

    # Make sure we don't have duplicates
    all_descriptors = list(set(all_descriptors))

    # RDKit Molecular Descriptor Calculator Class
    log.info("Computing RDKit Descriptors...")
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors)
    column_names = calc.GetDescriptorNames()
    descriptor_values = [calc.CalcDescriptors(m) for m in df["molecule"]]
    rdkit_features_df = pd.DataFrame(descriptor_values, columns=column_names)

    # Now compute Mordred Features
    log.info("Computing Mordred Descriptors...")
    descriptor_choice = [AcidBase, Aromatic, Polarizability, RotatableBond]
    calc = Calculator()
    for des in descriptor_choice:
        calc.register(des)
    mordred_df = calc.pandas(df["molecule"], nproc=1)

    # Combine the DataFrame with the RDKit and Mordred Descriptors added
    output_df = pd.concat([df, rdkit_features_df, mordred_df], axis=1)

    # Compute feature quality metrics
    feature_list = list(rdkit_features_df.columns) + list(mordred_df.columns)
    output_df = feature_quality_metrics(output_df, feature_list=feature_list)

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
        remove_mol_col (bool): Whether to drop the intermediate 'molecule' column. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with an additional 'canonical_smiles' column and,
                      optionally, the 'molecule' column.
    """
    # Identify the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'SMILES' column")

    # Convert SMILES to RDKit molecules
    df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # Handle invalid SMILES strings
    invalid_indices = df[df["molecule"].isna()].index
    if not invalid_indices.empty:
        log.critical(f"Invalid SMILES strings at indices: {invalid_indices.tolist()}")

    # Vectorized canonicalization
    def mol_to_canonical_smiles(mol):
        return Chem.MolToSmiles(mol) if mol else pd.NA

    df["canonical_smiles"] = df["molecule"].apply(mol_to_canonical_smiles)

    # Drop intermediate RDKit molecule column if requested
    if remove_mol_col:
        df.drop(columns=["molecule"], inplace=True)

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

    # Example custom logic: prioritize based on use-case specific criteria
    selected_tautomer = None
    highest_score = float("-inf")

    for taut in enumerated_tautomers:
        # Compute custom scoring logic:
        # 1. Prefer forms with fewer hydrogen bond donors (HBD) if membrane permeability is important
        # 2. Penalize forms with high molecular weight for better drug-likeness
        # 3. Incorporate known functional group preferences (e.g., keto > enol for binding)

        hbd = CalcNumHBD(taut)  # Hydrogen Bond Donors
        mw = CalcExactMolWt(taut)  # Molecular Weight
        aromatic_rings = taut.GetRingInfo().NumAromaticRings()  # Favor aromaticity

        # Example scoring: balance HBD, MW, and aromaticity
        score = -hbd - 0.01 * mw + aromatic_rings * 2

        # Update selected tautomer
        if score > highest_score:
            highest_score = score
            selected_tautomer = taut

    # Return the SMILES of the selected tautomer
    return Chem.MolToSmiles(selected_tautomer)


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
    # Standardize SMILES strings and create 'molecule' column for further processing
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
    df["tautomeric_form"] = df["molecule"].apply(safe_tautomerize)

    # Drop intermediate RDKit molecule column to clean up the DataFrame
    df.drop(columns=["molecule"], inplace=True)

    return df


if __name__ == "__main__":
    from workbench.api import DataSource

    # Pyridone molecule
    smiles = "C1=CC=NC(=O)C=C1"
    show(smiles)

    # SVG image of the molecule
    svg = svg_from_smiles(smiles)
    print(svg)

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

    # Test drug-likeness filter
    druglike_smiles = ["CC(C)=CCC\\C(C)=C/CO", "CC(C)CCCCCOC(=O)CCS", "OC(=O)CCCCCCCCC=C", "CC(C)(C)CCCCCC(=O)OC=C"]
    mols = [Chem.MolFromSmiles(smile) for smile in druglike_smiles]
    druglike = [is_druglike_compound(mol) for mol in mols]

    # Compute Molecular Descriptors
    df = pd.DataFrame({"smiles": [smiles, smiles, smiles, smiles, smiles]})
    df = compute_molecular_descriptors(df)
    print(df)

    df = DataSource("aqsol_data").pull_dataframe()[:1000]
    df = compute_molecular_descriptors(df)
    print(df)

    # Compute Morgan Fingerprints
    df = compute_morgan_fingerprints(df)
    print(df)

    # Perform Tautomerization
    df = perform_tautomerization(df)
    print(df)
