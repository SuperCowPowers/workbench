"""Local utilities for models that work with chemical information."""

import logging
import pandas as pd
from typing import List, Optional

# Molecular Descriptor Imports
from rdkit import Chem
from rdkit.Chem import Mol, Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
from rdkit.Chem import FunctionalGroups as FG
from mordred import Calculator
from mordred import AcidBase, Aromatic, Polarizability, RotatableBond

# Load functional group hierarchy once during initialization
fgroup_hierarchy = FG.BuildFuncGroupHierarchy()

# Set RDKit logger to only show errors or critical messages
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Set up the logger
log = logging.getLogger("workbench")


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


def halogen_toxicity_score(mol: Mol) -> (int, int):
    """
    Calculate the halogen count and toxicity threshold for a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Tuple[int, int]: (halogen_count, halogen_threshold), where the threshold
        scales with molecule size (minimum of 2 or 20% of atom count).
    """
    # Define halogens and count their occurrences
    halogens = {"Cl", "Br", "I", "F"}
    halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)

    # Define threshold: small molecules tolerate fewer halogens
    # Threshold scales with molecule size to account for reasonable substitution
    molecule_size = mol.GetNumAtoms()
    halogen_threshold = max(2, int(molecule_size * 0.2))  # Minimum 2, scaled by 20% of molecule size

    return halogen_count, halogen_threshold


def toxic_elements(mol: Mol) -> Optional[List[str]]:
    """
    Identifies toxic elements or specific forms of elements in a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Optional[List[str]]: List of toxic elements or specific forms if found, otherwise None.

    Notes:
        Halogen toxicity logic integrates with `halogen_toxicity_score` and scales thresholds
        based on molecule size.
    """
    # Always toxic elements (heavy metals and known toxic single elements)
    always_toxic = {"Pb", "Hg", "Cd", "As", "Be", "Tl", "Sb"}
    toxic_found = set()

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()

        # Check for always toxic elements
        if symbol in always_toxic:
            toxic_found.add(symbol)

        # Conditionally toxic nitrogen (positively charged)
        if symbol == "N" and formal_charge > 0:
            # Exclude benign quaternary ammonium (e.g., choline-like structures)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](C)(C)(C)C")):  # Example benign structure
                continue
            toxic_found.add("N+")

        # Halogen toxicity: Uses halogen_toxicity_score to flag excessive halogenation
        if symbol in {"Cl", "Br", "I", "F"}:
            halogen_count, halogen_threshold = halogen_toxicity_score(mol)
            if halogen_count > halogen_threshold:
                toxic_found.add(symbol)

    return list(toxic_found) if toxic_found else None


# Precompiled SMARTS patterns for custom toxic functional groups
toxic_smarts_patterns = [
    ("C(=S)N"),  # Dithiocarbamate
    ("P(=O)(O)(O)O"),  # Phosphate Ester
    ("[As](=O)(=O)-[OH]"),  # Arsenic Oxide
    ("[C](Cl)(Cl)(Cl)"),  # Trichloromethyl
    ("[Cr](=O)(=O)=O"),  # Chromium(VI)
    ("[N+](C)(C)(C)(C)"),  # Quaternary Ammonium
    ("[Se][Se]"),  # Diselenide
    ("c1c(Cl)c(Cl)c(Cl)c1"),  # Trichlorinated Aromatic Ring
    ("[CX3](=O)[CX4][Cl,Br,F,I]"),  # Halogenated Carbonyl
    ("[P+](C*)(C*)(C*)(C*)"),  # Phosphonium Group
    ("NC(=S)c1c(Cl)cccc1Cl"),  # Chlorobenzene Thiocarbamate
    ("NC(=S)Nc1ccccc1"),  # Phenyl Thiocarbamate
    ("S=C1NCCN1"),  # Thiourea Derivative
]
compiled_toxic_smarts = [Chem.MolFromSmarts(smarts) for smarts in toxic_smarts_patterns]

# Precompiled SMARTS patterns for exemptions
exempt_smarts_patterns = [
    "c1ccc(O)c(O)c1",  # Phenols
]
compiled_exempt_smarts = [Chem.MolFromSmarts(smarts) for smarts in exempt_smarts_patterns]


def toxic_groups(mol: Chem.Mol) -> Optional[List[str]]:
    """
    Check if a molecule contains known toxic functional groups using RDKit's functional groups and SMARTS patterns.

    Args:
        mol (rdkit.Chem.Mol): The molecule to evaluate.

    Returns:
        Optional[List[str]]: List of SMARTS patterns for toxic groups if found, otherwise None.
    """
    toxic_smarts_matches = []

    # Use RDKit's functional group definitions
    toxic_group_names = ["Nitro", "Azide", "Alcohol", "Aldehyde", "Halogen", "TerminalAlkyne"]
    for group_name in toxic_group_names:
        group_node = next(node for node in fgroup_hierarchy if node.label == group_name)
        if mol.HasSubstructMatch(Chem.MolFromSmarts(group_node.smarts)):
            toxic_smarts_matches.append(group_node.smarts)  # Use group_node's SMARTS directly

    # Check for custom precompiled toxic SMARTS patterns
    for smarts, compiled in zip(toxic_smarts_patterns, compiled_toxic_smarts):
        if mol.HasSubstructMatch(compiled):  # Use precompiled SMARTS
            toxic_smarts_matches.append(smarts)

    # Special handling for N+
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+]")):
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("C[N+](C)(C)C")):  # Exclude benign
            toxic_smarts_matches.append("[N+]")  # Append as SMARTS

    # Exempt stabilizing functional groups using precompiled patterns
    for compiled in compiled_exempt_smarts:
        if mol.HasSubstructMatch(compiled):
            return None

    return toxic_smarts_matches if toxic_smarts_matches else None


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
    df["meta"] = [{} for _ in range(len(df))]

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

        # Check for heavy metals
        if contains_heavy_metals(mol):
            tags.append("heavy_metals")

        # Check for toxic elements
        te = toxic_elements(mol)
        if te:
            tags.append("toxic_element")
            df.at[idx, "meta"]["toxic_elements"] = te

        # Check for toxic groups
        tg = toxic_groups(mol)
        if tg:
            tags.append("toxic_group")
            df.at[idx, "meta"]["toxic_groups"] = tg

        # Check for metalloenzyme-relevant metals
        if contains_metalloenzyme_relevant_metals(mol):
            tags.append("metalloenzyme")

        # Check for drug-likeness
        if is_druglike_compound(mol):
            tags.append("druglike")

        # Update tags
        df.at[idx, "tags"] = tags

    return df


def compute_molecular_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and add all the Molecular Descriptors

    Args:
        df(pd.DataFrame): The DataFrame to process and generate RDKit/Mordred Descriptors

    Returns:
        pd.DataFrame: The input DataFrame with all the RDKit Descriptors added
    """
    delete_mol_column = False

    # Check for the smiles column (any capitalization)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Compute/add all the Molecular Descriptors
    log.info("Computing Molecular Descriptors...")

    # Convert SMILES to RDKit molecule objects (vectorized)
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        delete_mol_column = True
        df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # If we have fragments in our compounds, get the largest fragment before computing descriptors
    largest_frags = df["molecule"].apply(remove_disconnected_fragments)

    # Now get all the RDKIT Descriptors
    all_descriptors = [x[0] for x in Descriptors._descList]

    # Make sure we don't have duplicates
    all_descriptors = list(set(all_descriptors))

    # RDKit Molecular Descriptor Calculator Class
    log.info("Computing RDKit Descriptors...")
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors)
    column_names = calc.GetDescriptorNames()
    descriptor_values = [calc.CalcDescriptors(m) for m in largest_frags]
    rdkit_features_df = pd.DataFrame(descriptor_values, columns=column_names)

    # Now compute Mordred Features
    log.info("Computing Mordred Descriptors...")
    descriptor_choice = [AcidBase, Aromatic, Polarizability, RotatableBond]
    calc = Calculator()
    for des in descriptor_choice:
        calc.register(des)
    mordred_df = calc.pandas(largest_frags, nproc=1)

    # Combine the DataFrame with the RDKit and Mordred Descriptors added
    # Note: This will overwrite any existing columns with the same name. This is a good thing
    #       since we want computed descriptors to overwrite anything in the input dataframe
    output_df = mordred_df.combine_first(rdkit_features_df).combine_first(df)

    # Lowercase all column names and ensure no duplicate column names
    output_df.columns = output_df.columns.str.lower()
    output_df = output_df.loc[:, ~output_df.columns.duplicated()]

    # Drop the intermediate 'molecule' column if it was added
    if delete_mol_column:
        del output_df["molecule"]

    # Return the DataFrame with the RDKit and Mordred Descriptors added
    return output_df


def compute_morgan_fingerprints(df: pd.DataFrame, radius=2, n_bits=2048, counts=True) -> pd.DataFrame:
    """Compute and add Morgan fingerprints to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        radius (int): Radius for the Morgan fingerprint.
        n_bits (int): Number of bits for the fingerprint.
        counts (bool): Count simulation for the fingerprint.

    Returns:
        pd.DataFrame: The input DataFrame with the Morgan fingerprints added as bit strings.

    Note:
        See: https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html
    """
    delete_mol_column = False

    # Check for the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Sanity check the molecule column (sometimes it gets serialized, which doesn't work)
    if "molecule" in df.columns and df["molecule"].dtype == "string":
        log.warning("Detected serialized molecules in 'molecule' column. Removing...")
        del df["molecule"]

    # Convert SMILES to RDKit molecule objects (vectorized)
    if "molecule" not in df.columns:
        log.info("Converting SMILES to RDKit Molecules...")
        delete_mol_column = True
        df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # If we have fragments in our compounds, get the largest fragment before computing fingerprints
    largest_frags = df["molecule"].apply(remove_disconnected_fragments)

    # Create a Morgan fingerprint generator
    if counts:
        n_bits *= 4  # Multiply by 4 to simulate counts
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, countSimulation=counts)

    # Compute Morgan fingerprints (vectorized)
    fingerprints = largest_frags.apply(
        lambda mol: (morgan_generator.GetFingerprint(mol).ToBitString() if mol else pd.NA)
    )

    # Add the fingerprints to the DataFrame
    df["morgan_fingerprint"] = fingerprints

    # Drop the intermediate 'molecule' column if it was added
    if delete_mol_column:
        del df["molecule"]
    return df


if __name__ == "__main__":

    # Small set of tests
    smiles = "O=C(CCl)c1ccc(Cl)cc1Cl"
    mol = Chem.MolFromSmiles(smiles)

    # Compute Molecular Descriptors
    df = pd.DataFrame({"smiles": [smiles, smiles, smiles, smiles, smiles]})
    df = compute_molecular_descriptors(df)
    print(df)

    # Compute Morgan Fingerprints
    df = compute_morgan_fingerprints(df)
    print(df)
