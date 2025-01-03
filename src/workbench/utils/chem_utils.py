"""Chem/RDKIT/Mordred utilities for Workbench"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
import base64
from sklearn.manifold import TSNE

# Try importing UMAP
try:
    import umap
except ImportError:
    umap = None

# Workbench Imports
from workbench.utils.pandas_utils import feature_quality_metrics
from workbench.utils.color_utils import is_dark, rgba_to_tuple

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


def img_from_smiles(
    smiles: str, width: int = 500, height: int = 500, background: str = "rgba(64, 64, 64, 1)"
) -> Optional[str]:
    """
    Generate an image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.
        width (int): Width of the image in pixels. Default is 500.
        height (int): Height of the image in pixels. Default is 500.
        background (str): Background color of the image. Default is dark grey

    Returns:
        str: PIL image of the molecule or None if the SMILES string is invalid.
    """

    # Set up the drawing options
    dos = Draw.MolDrawOptions()
    if is_dark(background):
        rdMolDraw2D.SetDarkMode(dos)
    dos.setBackgroundColour(rgba_to_tuple(background))

    # Convert the SMILES string to an RDKit molecule and generate the image
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, options=dos, size=(width, height))
        return img
    else:
        log.warning(f"Invalid SMILES: {smiles}")
        return None


def svg_from_smiles(
    smiles: str, width: int = 500, height: int = 500, background: str = "rgba(64, 64, 64, 1)"
) -> Optional[str]:
    """
    Generate an SVG image of the molecule represented by the given SMILES string.

    Args:
        smiles (str): A SMILES string representing the molecule.
        width (int): Width of the image in pixels. Default is 500.
        height (int): Height of the image in pixels. Default is 500.
        background (str): Background color of the image. Default is dark grey.

    Returns:
        Optional[str]: Encoded SVG string of the molecule or None if the SMILES string is invalid.
    """
    # Convert the SMILES string to an RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Compute 2D coordinates for the molecule
    AllChem.Compute2DCoords(mol)

    # Initialize the SVG drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)

    # Configure drawing options
    options = drawer.drawOptions()
    if is_dark(background):
        rdMolDraw2D.SetDarkMode(options)
    options.setBackgroundColour(rgba_to_tuple(background))

    # Draw the molecule
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Clean and encode the SVG
    svg = drawer.GetDrawingText()
    encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded_svg}"


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
    Remove disconnected fragments from a molecule, keeping the fragment with the most heavy atoms.

    Args:
        mol (Mol): RDKit molecule object.

    Returns:
        Optional[Mol]: The fragment with the most heavy atoms, or None if no such fragment exists.
    """
    # Get all fragments as individual molecules
    fragments = Chem.GetMolFrags(mol, asMols=True)

    # Return the fragment with the most heavy atoms, or None if no fragments
    if fragments:
        return max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
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
    Check if a molecule contains toxic elements or excessive halogenation.
    """
    always_toxic = {"Pb", "Hg", "Cd", "As", "Be", "Tl", "Sb"}
    conditional_toxic = {"Se", "Cr"}  # Now separate halogen logic below

    halogens = {"F", "Cl", "Br", "I"}
    halogen_count = sum(atom.GetSymbol() in halogens for atom in mol.GetAtoms())

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in always_toxic:
            return True
        if symbol in conditional_toxic:
            if symbol == "Cr" and atom.GetFormalCharge() == 6:  # Chromium(VI)
                return True
            if symbol == "Se" and atom.GetFormalCharge() != 0:  # Charged selenium
                return True

    # Generic halogen threshold check (e.g., excessive polyhalogenation)
    if halogen_count > 2:
        return True

    return False


def contains_toxic_groups(mol):
    """
    Check if a molecule contains known toxic groups via SMARTS patterns.
    """
    toxic_smarts = [
        "[Cr](=O)(=O)=O",          # Chromium(VI)
        "[As](=O)(=O)-[OH]",       # Arsenic oxide
        "[Hg]",                    # Mercury
        "[Pb]",                    # Lead
        "[Se][Se]",               # Diselenide
        "[Be]",                    # Beryllium
        "[Tl+]",                   # Thallium ion
        "c1(c(F)cc(F)c1)F",        # Example polychloro/fluoro phenols
        "[C](F)(F)F",              # Trifluoromethyl group
        "C#N",                     # Cyanides
        "[N+](=O)[O-]",            # Nitro groups
        "P(=O)(O)(O)O",            # Phosphate esters
        "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Example polycyclic aromatic
    ]

    for pattern in toxic_smarts:
        smarts = Chem.MolFromSmarts(pattern)
        if smarts and mol.HasSubstructMatch(smarts):
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


def add_compound_tags(df, mol_column="molecule"):
    """
    Adds a 'tags' column to a DataFrame, tagging compounds based on their properties.

    Args:
        df (pd.DataFrame): Input DataFrame containing molecular data.
        mol_column (str): Column name containing RDKit molecule objects.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'tags' column.
    """
    # Initialize the tags column
    df["tags"] = [[] for _ in range(len(df))]

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
                df.at[idx, "tags"] = tags
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
        df.at[idx, "tags"] = tags
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

    # Convert SMILES to RDKit molecule objects (vectorized)
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

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


def compute_morgan_fingerprints(df: pd.DataFrame, radius=2, nBits=4096) -> pd.DataFrame:
    """Compute and add Morgan fingerprints to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        radius (int): Radius for the Morgan fingerprint.
        nBits (int): Number of bits for the fingerprint.

    Returns:
        pd.DataFrame: The input DataFrame with the Morgan fingerprints added as bit strings.

    Note:
        See: https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html
    """

    # Check for the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Convert SMILES to RDKit molecule objects (vectorized)
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # Add Compound Tags
    df = add_compound_tags(df, mol_column="molecule")

    # Create a Morgan fingerprint generator
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits, countSimulation=True)

    # Compute Morgan fingerprints (vectorized)
    fingerprints = df["molecule"].apply(
        lambda mol: (morgan_generator.GetFingerprint(mol).ToBitString() if mol else pd.NA)
    )

    # Add the fingerprints to the DataFrame
    df["morgan_fingerprint"] = fingerprints
    return df


def project_fingerprints(df: pd.DataFrame, projection: str = "UMAP") -> pd.DataFrame:
    """Project fingerprints onto a 2D plane using dimensionality reduction techniques.

    Args:
        df (pd.DataFrame): Input DataFrame containing fingerprint data.
        projection (str): Dimensionality reduction technique to use (TSNE or UMAP).

    Returns:
        pd.DataFrame: The input DataFrame with the projected coordinates added as 'x' and 'y' columns.
    """
    # Check for the fingerprint column (case-insensitive)
    fingerprint_column = next((col for col in df.columns if "fingerprint" in col.lower()), None)
    if fingerprint_column is None:
        raise ValueError("Input DataFrame must have a fingerprint column")

    # Convert the bitstring fingerprint into a NumPy array
    df["fingerprint_bits"] = df[fingerprint_column].apply(
        lambda fp: np.array([int(bit) for bit in fp], dtype=np.float32)
    )

    # Create a matrix of fingerprints
    X = np.vstack(df["fingerprint_bits"].values)

    # Check for UMAP availability
    if projection == "UMAP" and umap is None:
        log.warning("UMAP is not available. Using TSNE instead.")
        projection = "TSNE"

    # Run the projection
    if projection == "TSNE":
        # Run TSNE on the fingerprint matrix
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding = tsne.fit_transform(X)
    else:
        # Run UMAP
        reducer = umap.UMAP(densmap=True)
        embedding = reducer.fit_transform(X)

    # Add coordinates to DataFrame
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]

    # Jitter
    jitter_scale = 0.1
    df["x"] += np.random.normal(0, jitter_scale, len(df))
    df["y"] += np.random.normal(0, jitter_scale, len(df))

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

    # PIL image of the molecule
    img = img_from_smiles(smiles)
    print(type(img))

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

    # Test compound tags
    df["molecule"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = add_compound_tags(df)

    # Compute Molecular Descriptors
    df = compute_molecular_descriptors(df)
    print(df)

    # Compute Morgan Fingerprints
    df = compute_morgan_fingerprints(df)
    print(df)

    # Project Fingerprints
    df = project_fingerprints(df, projection="UMAP")
    print(df)

    # Perform Tautomerization
    df = perform_tautomerization(df)
    print(df)
