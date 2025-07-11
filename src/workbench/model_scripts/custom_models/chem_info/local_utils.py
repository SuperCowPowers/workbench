"""Local utilities for models that work with chemical information."""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional

# Molecular Descriptor Imports
from rdkit import Chem
from rdkit.Chem import Mol, Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcExactMolWt
from rdkit import RDLogger
from rdkit.Chem import FunctionalGroups as FG
from mordred import Calculator as MordredCalculator
from mordred import AcidBase, Aromatic, Polarizability, RotatableBond

# Load functional group hierarchy once during initialization
fgroup_hierarchy = FG.BuildFuncGroupHierarchy()

# Set RDKit logger to only show errors or critical messages
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Set up the logger
log = logging.getLogger("workbench")


def remove_disconnected_fragments(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove disconnected fragments from a molecule, keeping the fragment with the most heavy atoms.

    Args:
        mol (Mol): RDKit molecule object.

    Returns:
        Mol: The fragment with the most heavy atoms, or None if no such fragment exists.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True)
    return max(fragments, key=lambda frag: frag.GetNumHeavyAtoms()) if fragments else None


def contains_heavy_metals(mol: Mol) -> bool:
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


def contains_metalloenzyme_relevant_metals(mol: Mol) -> bool:
    """
    Check if a molecule contains metals relevant to metalloenzymes.

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if metalloenzyme-relevant metals are detected, False otherwise.
    """
    metalloenzyme_metals = {"Zn", "Cu", "Fe", "Mn", "Co"}
    return any(atom.GetSymbol() in metalloenzyme_metals for atom in mol.GetAtoms())


def contains_salts(mol: Mol) -> bool:
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


def add_compound_tags(df, mol_column="molecule") -> pd.DataFrame:
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


def compute_molecular_descriptors(df: pd.DataFrame, tautomerize=True) -> pd.DataFrame:
    """Compute and add all the Molecular Descriptors

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        tautomerize (bool): Whether to tautomerize the SMILES strings.

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
    log.info("Converting SMILES to RDKit Molecules...")
    df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # Make sure our molecules are not None
    failed_smiles = df[df["molecule"].isnull()][smiles_column].tolist()
    if failed_smiles:
        log.error(f"Failed to convert the following SMILES to molecules: {failed_smiles}")
    df = df.dropna(subset=["molecule"])

    # If we have fragments in our compounds, get the largest fragment before computing descriptors
    df["molecule"] = df["molecule"].apply(remove_disconnected_fragments)

    # Tautomerize the molecules if requested
    if tautomerize:
        log.info("Tautomerizing molecules...")
        tautomer_enumerator = TautomerEnumerator()
        df["molecule"] = df["molecule"].apply(tautomer_enumerator.Canonicalize)

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
    descriptor_values = [calc.CalcDescriptors(m) for m in df["molecule"]]

    # Lowercase the column names
    column_names = [name.lower() for name in calc.GetDescriptorNames()]
    rdkit_features_df = pd.DataFrame(descriptor_values, columns=column_names)

    # Now compute Mordred Features
    log.info("Computing Mordred Descriptors...")
    descriptor_choice = [AcidBase, Aromatic, Polarizability, RotatableBond]
    calc = MordredCalculator()
    for des in descriptor_choice:
        calc.register(des)
    mordred_df = calc.pandas(df["molecule"], nproc=1)

    # Lowercase the column names
    mordred_df.columns = [col.lower() for col in mordred_df.columns]

    # Compute stereochemistry descriptors
    stereo_df = compute_stereochemistry_descriptors(df)

    # Combine the DataFrame with the RDKit and Mordred Descriptors added
    # Note: This will overwrite any existing columns with the same name. This is a good thing
    #       since we want computed descriptors to overwrite anything in the input dataframe
    output_df = stereo_df.combine_first(mordred_df).combine_first(rdkit_features_df)

    # Ensure no duplicate column names
    output_df = output_df.loc[:, ~output_df.columns.duplicated()]

    # Reorder the columns to have all the ones in the input df first and then the descriptors
    input_columns = df.columns.tolist()
    output_df = output_df[input_columns + [col for col in output_df.columns if col not in input_columns]]

    # Drop the intermediate 'molecule' column
    del output_df["molecule"]

    # Return the DataFrame with the RDKit and Mordred Descriptors added
    return output_df


def compute_stereochemistry_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stereochemistry descriptors for molecules in a DataFrame.

    This function analyzes the stereochemical properties of molecules, including:
    - Chiral centers (R/S configuration)
    - Double bond stereochemistry (E/Z configuration)

    Args:
        df (pd.DataFrame): Input DataFrame with RDKit molecule objects in 'molecule' column

    Returns:
        pd.DataFrame: DataFrame with added stereochemistry descriptors
    """
    if "molecule" not in df.columns:
        raise ValueError("Input DataFrame must have a 'molecule' column")

    log.info("Computing stereochemistry descriptors...")
    output_df = df.copy()

    # Create helper functions to process a single molecule
    def process_molecule(mol):
        if mol is None:
            log.warning("Found a None molecule, skipping...")
            return {
                "chiral_centers": 0,
                "r_cnt": 0,
                "s_cnt": 0,
                "db_stereo": 0,
                "e_cnt": 0,
                "z_cnt": 0,
                "chiral_fp": 0,
                "db_fp": 0,
            }

        try:
            # Use the more accurate CIP labeling algorithm (Cahn-Ingold-Prelog rules)
            # This assigns R/S to chiral centers and E/Z to double bonds based on
            # the priority of substituents (atomic number, mass, etc.)
            rdCIPLabeler.AssignCIPLabels(mol)

            # Find all potential stereochemistry sites in the molecule
            stereo_info = Chem.FindPotentialStereo(mol)

            # Initialize counters
            specified_centers = 0  # Number of chiral centers with defined stereochemistry
            r_cnt = 0  # Count of R configured centers
            s_cnt = 0  # Count of S configured centers
            stereo_atoms = []  # List to store atom indices and their R/S configuration

            specified_bonds = 0  # Number of double bonds with defined stereochemistry
            e_cnt = 0  # Count of E (trans) configured double bonds
            z_cnt = 0  # Count of Z (cis) configured double bonds
            stereo_bonds = []  # List to store bond indices and their E/Z configuration

            # Process all stereo information found in the molecule
            for element in stereo_info:
                # Handle tetrahedral chiral centers
                if element.type == Chem.StereoType.Atom_Tetrahedral:
                    atom_idx = element.centeredOn

                    # Only count centers where stereochemistry is explicitly defined
                    if element.specified == Chem.StereoSpecified.Specified:
                        specified_centers += 1
                        if element.descriptor == Chem.StereoDescriptor.Tet_CCW:
                            r_cnt += 1
                            stereo_atoms.append((atom_idx, "R"))
                        elif element.descriptor == Chem.StereoDescriptor.Tet_CW:
                            s_cnt += 1
                            stereo_atoms.append((atom_idx, "S"))

                # Handle double bond stereochemistry
                elif element.type == Chem.StereoType.Bond_Double:
                    bond_idx = element.centeredOn

                    # Only count bonds where stereochemistry is explicitly defined
                    if element.specified == Chem.StereoSpecified.Specified:
                        specified_bonds += 1
                        if element.descriptor == Chem.StereoDescriptor.Bond_Trans:
                            e_cnt += 1
                            stereo_bonds.append((bond_idx, "E"))
                        elif element.descriptor == Chem.StereoDescriptor.Bond_Cis:
                            z_cnt += 1
                            stereo_bonds.append((bond_idx, "Z"))

            # Calculate chiral center fingerprint - unique bit vector for stereochemical configuration
            chiral_fp = 0
            if stereo_atoms:
                for i, (idx, stereo) in enumerate(sorted(stereo_atoms, key=lambda x: x[0])):
                    bit_val = 1 if stereo == "R" else 0
                    chiral_fp += bit_val << i  # Shift bits to create a unique fingerprint

            # Calculate double bond fingerprint - bit vector for E/Z configurations
            db_fp = 0
            if stereo_bonds:
                for i, (idx, stereo) in enumerate(sorted(stereo_bonds, key=lambda x: x[0])):
                    bit_val = 1 if stereo == "E" else 0
                    db_fp += bit_val << i  # Shift bits to create a unique fingerprint

            return {
                "chiral_centers": specified_centers,
                "r_cnt": r_cnt,
                "s_cnt": s_cnt,
                "db_stereo": specified_bonds,
                "e_cnt": e_cnt,
                "z_cnt": z_cnt,
                "chiral_fp": chiral_fp,
                "db_fp": db_fp,
            }

        except Exception as e:
            log.warning(f"Error processing stereochemistry: {str(e)}")
            return {
                "chiral_centers": 0,
                "r_cnt": 0,
                "s_cnt": 0,
                "db_stereo": 0,
                "e_cnt": 0,
                "z_cnt": 0,
                "chiral_fp": 0,
                "db_fp": 0,
            }

    # Process all molecules and collect results
    results = []
    for mol in df["molecule"]:
        results.append(process_molecule(mol))

    # Add all descriptors to the output dataframe
    for key in results[0].keys():
        output_df[key] = [r[key] for r in results]

    # Boolean flag indicating if the molecule has any stereochemistry defined
    output_df["has_stereo"] = (output_df["chiral_centers"] > 0) | (output_df["db_stereo"] > 0)

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
        # Make sure our molecules are not None
        failed_smiles = df[df["molecule"].isnull()][smiles_column].tolist()
        if failed_smiles:
            log.error(f"Failed to convert the following SMILES to molecules: {failed_smiles}")
        df = df.dropna(subset=["molecule"])

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
    df["fingerprint"] = fingerprints

    # Drop the intermediate 'molecule' column if it was added
    if delete_mol_column:
        del df["molecule"]
    return df


def fingerprints_to_matrix(fingerprints, dtype=np.uint8):
    """
    Convert bitstring fingerprints to numpy matrix.

    Args:
        fingerprints: pandas Series or list of bitstring fingerprints
        dtype: numpy data type (uint8 is default: np.bool_ is good for Jaccard computations

    Returns:
        dense numpy array of shape (n_molecules, n_bits)
    """

    # Dense matrix representation (we might support sparse in the future)
    return np.array([list(fp) for fp in fingerprints], dtype=dtype)


def canonicalize(df: pd.DataFrame, remove_mol_col: bool = True) -> pd.DataFrame:
    """
    Generate RDKit's canonical SMILES for each molecule in the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a column named 'SMILES' (case-insensitive).
        remove_mol_col (bool): Whether to drop the intermediate 'molecule' column. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with an additional 'smiles_canonical' column and,
                      optionally, the 'molecule' column.
    """
    # Identify the SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'SMILES' column")

    # Convert SMILES to RDKit molecules
    df["molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # Log invalid SMILES
    invalid_indices = df[df["molecule"].isna()].index
    if not invalid_indices.empty:
        log.critical(f"Invalid SMILES strings at indices: {invalid_indices.tolist()}")

    # Drop rows where SMILES failed to convert to molecule
    df.dropna(subset=["molecule"], inplace=True)

    # Remove disconnected fragments (keep the largest fragment)
    df["molecule"] = df["molecule"].apply(lambda mol: remove_disconnected_fragments(mol) if mol else None)

    # Convert molecules to canonical SMILES (preserving isomeric information)
    df["smiles_canonical"] = df["molecule"].apply(
        lambda mol: Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
    )

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


def tautomerize_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform tautomer enumeration and canonicalization on a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.

    Returns:
        pd.DataFrame: A new DataFrame with additional 'smiles_canonical' and 'smiles_tautomer' columns.
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
    df["smiles_tautomer"] = df["molecule"].apply(safe_tautomerize)

    # Drop intermediate RDKit molecule column to clean up the DataFrame
    df.drop(columns=["molecule"], inplace=True)

    # Now switch the smiles columns
    df.rename(columns={"smiles": "smiles_orig", "smiles_tautomer": "smiles"}, inplace=True)

    return df


if __name__ == "__main__":

    # Small set of tests
    smiles = "O=C(CCl)c1ccc(Cl)cc1Cl"
    mol = Chem.MolFromSmiles(smiles)

    # Compute Molecular Descriptors
    df = pd.DataFrame({"smiles": [smiles, smiles, smiles, smiles, smiles]})
    md_df = compute_molecular_descriptors(df)
    print(md_df)

    # Compute Morgan Fingerprints
    fp_df = compute_morgan_fingerprints(df)
    print(fp_df)

    # Perform Tautomerization
    t_df = tautomerize_smiles(df)
    print(t_df)
