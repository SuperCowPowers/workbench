"""Chem/RDKIT/Mordred utilities for Workbench"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import base64
from sklearn.manifold import TSNE

# Try importing UMAP
try:
    import umap
except ImportError:
    umap = None

# Workbench Imports
from workbench.utils.color_utils import is_dark, rgba_to_tuple

# Molecular Descriptor Imports
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, Descriptors, rdFingerprintGenerator, Draw, SDWriter
from rdkit.Chem.Draw import rdMolDraw2D
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

"""FIXME:
Let's figure out what we're not just using these RDKit methods

from rdkit.Chem import PandasTools
df = PandasTools.LoadSDF('file.sdf', molColName='ROMol', smilesName='SMILES')
PandasTools.WriteSDF(df, 'file.sdf', molColName='ROMol', properties=list(df.columns))

"""


def df_to_sdf_file(
    df: pd.DataFrame,
    output_file: str,
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    skip_invalid: bool = True,
    generate_3d: bool = True,
):
    """
    Convert DataFrame with SMILES to SDF file.

    Args:
        df: DataFrame containing SMILES and other data
        output_file: Path to output SDF file
        smiles_col: Column name containing SMILES strings
        id_col: Column to use as molecule ID/name
        include_cols: Specific columns to include as properties (default: all except smiles and molecule columns)
        skip_invalid: Skip invalid SMILES instead of raising error
        generate_3d: Generate 3D coordinates and optimize geometry
    """
    written_count = 0

    with SDWriter(output_file) as writer:
        writer.SetForceV3000(True)
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                if not skip_invalid:
                    raise ValueError(f"Invalid SMILES at row {idx}: {row[smiles_col]}")
                continue

            # Generate 3D coordinates
            if generate_3d:
                mol = Chem.AddHs(mol)

                # Try progressively more aggressive embedding strategies
                embed_strategies = [
                    {"maxAttempts": 1000, "randomSeed": 42},
                    {"maxAttempts": 1000, "randomSeed": 42, "useRandomCoords": True},
                    {"maxAttempts": 1000, "randomSeed": 42, "boxSizeMult": 5.0},
                ]

                embedded = False
                for strategy in embed_strategies:
                    if AllChem.EmbedMolecule(mol, **strategy) != -1:
                        embedded = True
                        break

                if not embedded:
                    if not skip_invalid:
                        raise ValueError(f"Could not generate 3D coords for row {idx}")
                    continue

                AllChem.MMFFOptimizeMolecule(mol)

            # Set molecule name/ID
            if id_col and id_col in df.columns:
                mol.SetProp("_Name", str(row[id_col]))

            # Determine which columns to include
            if include_cols:
                cols_to_add = [col for col in include_cols if col in df.columns and col != smiles_col]
            else:
                # Auto-exclude common molecule column names and SMILES column
                mol_col_names = ["mol", "molecule", "rdkit_mol", "Mol"]
                cols_to_add = [col for col in df.columns if col != smiles_col and col not in mol_col_names]

            # Add properties
            for col in cols_to_add:
                mol.SetProp(col, str(row[col]))

            writer.write(mol)
            written_count += 1

    log.important(f"Wrote {written_count} molecules to SDF: {output_file}")


def sdf_file_to_df(
    sdf_file: str,
    include_smiles: bool = True,
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
    include_props: Optional[List[str]] = None,
    exclude_props: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert SDF file to DataFrame.

    Args:
        sdf_file: Path to input SDF file
        include_smiles: Add SMILES column to output
        smiles_col: Name for SMILES column
        id_col: Column name for molecule ID/name (uses _Name property)
        include_props: Specific properties to include (default: all)
        exclude_props: Properties to exclude from output

    Returns:
        DataFrame with molecules and their properties
    """
    data = []

    suppl = Chem.SDMolSupplier(sdf_file)
    for idx, mol in enumerate(suppl):
        if mol is None:
            log.warning(f"Could not parse molecule at index {idx}")
            continue

        row_data = {}

        # Add SMILES if requested
        if include_smiles:
            row_data[smiles_col] = Chem.MolToSmiles(mol)

        # Add molecule name/ID if requested
        if id_col and mol.HasProp("_Name"):
            row_data[id_col] = mol.GetProp("_Name")

        # Get all properties
        prop_names = mol.GetPropNames()

        # Filter properties based on include/exclude lists
        if include_props:
            prop_names = [p for p in prop_names if p in include_props]
        if exclude_props:
            prop_names = [p for p in prop_names if p not in exclude_props]

        # Add properties to row
        for prop in prop_names:
            if prop != "_Name":  # Skip _Name if we already handled it
                row_data[prop] = mol.GetProp(prop)

        data.append(row_data)

    df = pd.DataFrame(data)
    log.important(f"Read {len(df)} molecules from SDF: {sdf_file}")

    return df


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


def geometric_mean(series: pd.Series) -> float:
    """Computes the geometric mean manually to avoid using scipy."""
    return np.exp(np.log(series).mean())


def rollup_experimental_data(
    df: pd.DataFrame, id: str, time: str, target: str, use_gmean: bool = False
) -> pd.DataFrame:
    """
    Rolls up a dataset by selecting the largest time per unique ID and averaging the target value
    if multiple records exist at that time. Supports both arithmetic and geometric mean.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        id (str): Column representing the unique molecule ID.
        time (str): Column representing the time.
        target (str): Column representing the target value.
        use_gmean (bool): Whether to use the geometric mean instead of the arithmetic mean.

    Returns:
        pd.DataFrame: Rolled-up dataframe with all original columns retained.
    """
    # Find the max time per unique ID
    max_time_df = df.groupby(id)[time].transform("max")
    filtered_df = df[df[time] == max_time_df]

    # Define aggregation function
    agg_func = geometric_mean if use_gmean else np.mean

    # Perform aggregation on all columns
    agg_dict = {col: "first" for col in df.columns if col not in [target, id, time]}
    agg_dict[target] = lambda x: agg_func(x) if len(x) > 1 else x.iloc[0]  # Apply mean or gmean

    rolled_up_df = filtered_df.groupby([id, time]).agg(agg_dict).reset_index()
    return rolled_up_df


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

    # Create a matrix of fingerprints
    X = fingerprints_to_matrix(df[fingerprint_column])

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
        # reducer = umap.UMAP(densmap=True)
        reducer = umap.UMAP(metric="jaccard")
        embedding = reducer.fit_transform(X)

    # Add coordinates to DataFrame
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]

    # If vertices disconnect from the manifold, they are given NaN values (so replace with 0)
    df["x"] = df["x"].fillna(0)
    df["y"] = df["y"].fillna(0)

    # Jitter
    jitter_scale = 0.1
    df["x"] += np.random.uniform(0, jitter_scale, len(df))
    df["y"] += np.random.uniform(0, jitter_scale, len(df))

    return df


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


def _get_salt_feature_columns() -> List[str]:
    """Internal: Return list of all salt feature column names"""
    return [
        "has_salt",
        "mw_ratio",
        "salt_to_api_ratio",
        "has_metal_salt",
        "has_halide",
        "ionic_strength_proxy",
        "has_organic_salt",
    ]


def _classify_salt_types(salt_frags: List[Chem.Mol]) -> Dict[str, int]:
    """Internal: Classify salt fragments into categories"""
    features = {
        "has_organic_salt": 0,
        "has_metal_salt": 0,
        "has_halide": 0,
    }

    for frag in salt_frags:
        # Get atoms
        atoms = [atom.GetSymbol() for atom in frag.GetAtoms()]

        # Metal detection
        metals = ["Na", "K", "Ca", "Mg", "Li", "Zn", "Fe", "Al"]
        if any(metal in atoms for metal in metals):
            features["has_metal_salt"] = 1

        # Halide detection
        halides = ["Cl", "Br", "I", "F"]
        if any(halide in atoms for halide in halides):
            features["has_halide"] = 1

        # Organic vs inorganic (simple heuristic: contains C)
        if "C" in atoms:
            features["has_organic_salt"] = 1

    return features


def extract_advanced_salt_features(
    mol: Optional[Chem.Mol],
) -> Tuple[Optional[Dict[str, Union[int, float]]], Optional[Chem.Mol]]:
    """Extract comprehensive salt-related features from RDKit molecule"""
    if mol is None:
        return None, None

    # Get fragments
    fragments = Chem.GetMolFrags(mol, asMols=True)

    # Identify API (largest organic fragment) vs salt fragments
    fragment_weights = [(frag, Descriptors.MolWt(frag)) for frag in fragments]
    fragment_weights.sort(key=lambda x: x[1], reverse=True)

    # Find largest organic fragment as API
    api_mol = None
    salt_frags = []

    for frag, mw in fragment_weights:
        atoms = [atom.GetSymbol() for atom in frag.GetAtoms()]
        if "C" in atoms and api_mol is None:  # First organic fragment = API
            api_mol = frag
        else:
            salt_frags.append(frag)

    # Fallback: if no organic fragments, use largest
    if api_mol is None:
        api_mol = fragment_weights[0][0]
        salt_frags = [frag for frag, _ in fragment_weights[1:]]

    # Initialize all features with default values
    features = {col: 0 for col in _get_salt_feature_columns()}
    features["mw_ratio"] = 1.0  # default for no salt

    # Basic features
    features.update(
        {
            "has_salt": int(len(salt_frags) > 0),
            "mw_ratio": Descriptors.MolWt(api_mol) / Descriptors.MolWt(mol),
        }
    )

    if salt_frags:
        # Salt characterization
        total_salt_mw = sum(Descriptors.MolWt(frag) for frag in salt_frags)
        features.update(
            {
                "salt_to_api_ratio": total_salt_mw / Descriptors.MolWt(api_mol),
                "ionic_strength_proxy": sum(abs(Chem.GetFormalCharge(frag)) for frag in salt_frags),
            }
        )

        # Salt type classification
        features.update(_classify_salt_types(salt_frags))

    return features, api_mol


def add_salt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add salt features to dataframe with 'molecule' column containing RDKit molecules"""
    salt_features_list = []

    for idx, row in df.iterrows():
        mol = row["molecule"]
        features, clean_mol = extract_advanced_salt_features(mol)

        if features is None:
            # Handle invalid molecules
            features = {col: None for col in _get_salt_feature_columns()}

        salt_features_list.append(features)

    # Convert to DataFrame and concatenate
    salt_df = pd.DataFrame(salt_features_list)
    return pd.concat([df, salt_df], axis=1)


def feature_resolution_issues(df: pd.DataFrame, features: List[str], show_cols: Optional[List[str]] = None) -> None:
    """
    Identify and print groups in a DataFrame where the given features have more than one unique SMILES,
    sorted by group size (largest number of unique SMILES first).

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        features (List[str]): List of features to check.
        show_cols (Optional[List[str]]): Columns to display; defaults to all columns.
    """
    # Check for the 'smiles' column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    show_cols = show_cols if show_cols is not None else df.columns.tolist()

    # Drop duplicates to keep only unique SMILES for each feature combination
    unique_df = df.drop_duplicates(subset=[smiles_column] + features)

    # Find groups with more than one unique SMILES
    group_counts = unique_df.groupby(features).size()
    collision_groups = group_counts[group_counts > 1].sort_values(ascending=False)

    # Print each group in order of size (largest first)
    for group, count in collision_groups.items():
        # Get the rows for this group
        if isinstance(group, tuple):
            group_mask = (unique_df[features] == group).all(axis=1)
        else:
            group_mask = unique_df[features[0]] == group

        group_df = unique_df[group_mask]

        print(f"Feature Group (unique SMILES: {count}):")
        print(group_df[show_cols])
        print("\n")


if __name__ == "__main__":
    from workbench.api import DataSource

    # Set pandas display options
    pd.options.display.max_columns = 20
    pd.options.display.max_colwidth = 200
    pd.options.display.width = 1400

    # Test data
    # Create test molecules with known E/Z stereochemistry
    test_smiles = [
        # E (trans) examples
        "C/C=C/C",  # trans-2-butene
        "C/C=C/Cl",  # trans-2-chloro-2-butene
        "ClC=CCl",  # non-stereo notation
        "Cl/C=C/Cl",  # trans-1,2-dichloroethene
        # Z (cis) examples
        "C/C=C\\C",  # cis-2-butene
        "C/C=C\\Cl",  # cis-2-chloro-2-butene
        "Cl/C=C\\Cl",  # cis-1,2-dichloroethene
        # More complex examples
        "C/C=C/C=C",  # trans-2,4-hexadiene
        "C/C=C\\C=C",  # mix of cis and trans
        "C/C=C/C=C/C",  # all-trans-2,4,6-octatriene
        "C/C(Cl)=C\\C",  # substituted example
        # Non-stereochemical double bonds
        "C=C",  # ethene (no stereochemistry)
        "C=CC=C",  # 1,3-butadiene (no specified stereochemistry)
        "C1=CCCCC1",  # cyclohexene (no stereochemistry possible)
        # Compare with chiral centers
        "C[C@H](Cl)Br",  # chiral molecule
        "CC(Cl)Br"  # non-chiral notation
        "N[C@H]1CC[C@@H](CC1)[NH2+]CCF",  # From RDKIT/Github discussion example
    ]

    # AQSol Smiles
    aqsol_smiles = [
        r"CCCCCCCC\\C=C\\CCCCCCCCNCCCNCCCNCCCN",
        r"COC1=CC=C(C=C1N\\N=C1/C(=O)C(=CC2=CC=CC=C12)C(=O)NC1=CC(Cl)=CC=C1C)C(=O)NC1=CC=CC=C1",
        r"NC(=O)N\\N=C\\C(O)C(O)C(O)CO",
        r"C1=CC=C(C=C1)\\N=N\\C1=CC=CC=C1",
        r"CC(=O)N\\N=C\\C1=CC=C(O1)[N+]([O-])=O",
        r"CC(=O)OCCN(CCC#N)C1=CC=C(C=C1)\\N=N\\C1=CC=C(C=C1)[N+]([O-])=O",
        r"ClC1=CC=C(Cl)C(N\\N=C2/C(=O)C(=CC3=CC=CC=C23)C(=O)NC2=CC=C3NC(=O)NC3=C2)=C1",
        r"NC1=CC=C(C=C1)\\N=N\\C1=CC=CC=C1",
        r"OC(=O)\\C=C/C=C\\C(O)=O",
        r"CCOC(=O)\\C=C\\C1=CC=CC=C1",
        r"CC(=O)\\C=C\\C1=C(C)CCCC1(C)C",
        r"C\\C(=C/C(O)=O)C(O)=O",
        r"CCC\\C=C\\C",
        r"CC1=NN(C(=O)\\C1=N\\NC1=CC=C(C=C1Cl)C1=CC=C(N\\N=C2/C(C)=NN(C2=O)C2=CC=CC=C2)C(Cl)=C1)C1=CC=CC=C1",
        r"OC(C1=CC2C3C(C1\\C2=C(\\C1=CC=CC=C1)C1=CC=CC=N1)C(=O)NC3=O)(C1=CC=CC=C1)C1=CC=CC=N1",
        r"COC1=CC=C(\\C=C\\C(=O)C2=C(O)C=CC=C2)C=C1",
        r"CC\\C(=C(\\CC)C1=CC=C(O)C=C1)C1=CC=C(O)C=C1",
        r"C\\C=C\\OC1CCC(CC1)O\\C=C\\C",
        r"CC(C)=C[C@@H]1[C@@H](C(=O)O[C@H]2CC(=O)C(C\\C=C/C=C)=C2C)C1(C)C",
        r"CC\\C=C\\C",
        r"COC(=O)C(\\C)=C\\[C@@H]1[C@@H](C(=O)O[C@H]2CC(=O)C(C\\C=C/C=C)=C2C)C1(C)C",
        r"CC1=C(F)C(F)=C(COC(=O)C2C(\\C=C(/Cl)C(F)(F)F)C2(C)C)C(F)=C1F",
        r"CCC(=O)OC\\C=C(/C)\\C=C\\C=C(/C)\\C=C\\C1=C(C)CCCC1(C)C",
        r"CC(=O)C(\\C)=C/C1C(C)=CCCC1(C)C",
        r"CC(=O)C(\\N=N\\C1=CC=CC=C1C(O)=O)C(=O)NC1=CC=C2NC(=O)NC2=C1",
        r"O\\N=C1\\CCCC=C1",
        r"CCCCCCCCCCCCCCCC(=O)NCCCCCCCC\\C=C/CCCCCCCC",
        r"ClC\\C=C/CCl",
        r"CC(=O)C(\\N=N\\C1=CC=C(Cl)C=C1[N+]([O-])=O)C(=O)NC1=CC=C2NC(=O)NC2=C1",
        r"OC(=O)\\C=C(/Cl)C1=CC=CC=C1",
        r"CC(=O)C(\\N=N\\C1=CC=C(C=C1)[N+]([O-])=O)C(=O)NC1=CC=C2NC(=O)NC2=C1",
        r"CC\\C=C/CCO",
    ]
    all_smiles = test_smiles + aqsol_smiles

    # Create molecules
    mols = [Chem.MolFromSmiles(s) for s in all_smiles]

    # Create test dataframe
    df = pd.DataFrame({"smiles": all_smiles, "molecule": mols})

    # Test Stereochemistry Descriptors
    # See: https://github.com/rdkit/rdkit/discussions/6567
    df = compute_stereochemistry_descriptors(df)
    # Print all the columns except molecule
    print(df.drop(columns=["molecule"]))

    # Toxicity tests
    smiles = "O=C(CCl)c1ccc(Cl)cc1Cl"
    mol = Chem.MolFromSmiles(smiles)
    print(toxic_elements(mol))
    print(toxic_groups(mol))

    # Pyridone molecule
    smiles = "C1=CC=NC(=O)C=C1"
    show(smiles)

    # SVG image of the molecule
    svg = svg_from_smiles(smiles)

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

    # Test drug-likeness filter and print results
    druglike_smiles = ["CC(C)=CCC\\C(C)=C/CO", "CC(C)CCCCCOC(=O)CCS", "OC(=O)CCCCCCCCC=C", "CC(C)(C)CCCCCC(=O)OC=C"]
    mols = [Chem.MolFromSmiles(smile) for smile in druglike_smiles]
    druglike = [is_druglike_compound(mol) for mol in mols]

    for smile, is_druglike in zip(druglike_smiles, druglike):
        print(f"SMILES: {smile} -> Drug-like: {is_druglike}")

    # Test mol/None issue
    df = DataSource("aqsol_data").pull_dataframe()[:100]
    mol_df = compute_molecular_descriptors(df)

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

    # Debug a few compounds that have issues
    debug_smiles = {
        "id": ["B-1579", "B-1866"],
        "smiles": ["CC1=CC=C[NH++]([O-])[CH-]1", "OC(=O)C1=C[NH++]([O-])[CH-]C=C1"],
    }
    debug_df = pd.DataFrame(debug_smiles)
    debug_df = compute_morgan_fingerprints(debug_df)
    print(debug_df)

    # Project Fingerprints
    df = project_fingerprints(df, projection="UMAP")
    print(df)

    # Perform Tautomerization
    df = tautomerize_smiles(df)
    print(df)

    # Test Rollup Experimental Data
    test_data = {
        "id": ["1", "1", "2", "2", "3", "4", "4", "5", "5", "6", "6"],
        "time_hr": [1, 4, 3, 3, 2, np.nan, 5, 6, 6, np.nan, np.nan],
        "target_value": [1.90, 4.03, 2.5, 3.5, 7.8, 6.2, 8.1, np.nan, 5.4, 6.7, 6.9],
        "smiles": [
            "CC(=O)O",  # Acetic acid
            "CC(=O)O",
            "C1CCCCC1",  # Cyclohexane
            "C1CCCCC1",
            "C1=CC=CC=C1",  # Benzene
            "CCO",  # Ethanol
            "CCO",
            "CC(C)=O",  # Acetone
            "CC(C)=O",
            "CC(C)=O",
            "CC(C)=O",
        ],
    }

    # Create test DataFrame
    test_df = pd.DataFrame(test_data)
    print("Original Test DataFrame:")
    print(test_df)
    print("\n")

    # Test with arithmetic mean
    result_df = rollup_experimental_data(test_df, id="id", time="time_hr", target="target_value", use_gmean=False)
    print("Result with Arithmetic Mean:")
    print(result_df)
    print("\n")

    # Test with geometric mean
    result_df_gmean = rollup_experimental_data(test_df, id="id", time="time_hr", target="target_value", use_gmean=True)
    print("Result with Geometric Mean:")
    print(result_df_gmean)

    # Test some salted compounds
    test_data = {
        "id": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        "target_value": [1.90, 4.03, 2.5, 3.5, 7.8, 6.2, 8.1, 6.9, 5.4, 3.2, 4.8, 7.1],
        "smiles": [
            "CC(=O)O",  # Acetic acid (no salt)
            "C1CCCCC1",  # Cyclohexane (no salt)
            "C1=CC=CC=C1",  # Benzene (no salt)
            "CC(=O)O.[K+]",  # Potassium acetate (metal cation)
            "CC(=O)O.[Ca+2]",  # Calcium acetate (metal cation)
            "CC(=O)O.[Na+]",  # Sodium acetate (metal cation)
            "CCO.Cl",  # Ethanol hydrochloride (halide anion)
            "C1=CC=CC=C1.O.O",  # Benzene hydrate (inorganic)
            "CC(=O)[O-].C[NH3+]",  # Methylammonium acetate (organic anion + cation)
            "c1ccc(cc1)[NH3+].[Cl-]",  # Aniline HCl (organic cation, halide anion)
            "CC(=O)[O-].CC[NH3+]",  # Ethylammonium acetate (organic anion + cation)
            "CCO.[Br-].[Na+]",  # Multiple salt components
        ],
    }

    # Create test DataFrame
    test_df = pd.DataFrame(test_data)

    # Convert SMILES to molecules
    test_df["molecule"] = test_df["smiles"].apply(Chem.MolFromSmiles)

    # Test individual function
    print("Testing individual salt feature extraction:")
    for i, row in test_df.iterrows():
        if i < 3:  # Test first few
            features, clean_mol = extract_advanced_salt_features(row["molecule"])
            print(f"SMILES: {row['smiles']}")
            print(f"Features: {features}")
            print(f"Clean mol atoms: {clean_mol.GetNumAtoms() if clean_mol else 'None'}")
            print("---")

    # Test full DataFrame processing
    print("\nTesting DataFrame processing:")
    result_df = add_salt_features(test_df)

    # Display results focusing on salt-related columns
    salt_cols = _get_salt_feature_columns()
    display_cols = ["smiles"] + salt_cols
    print(result_df[display_cols].to_string())

    # Summary stats
    print(f"\nDataFrame shape before: {test_df.shape}")
    print(f"DataFrame shape after: {result_df.shape}")
    print(f"Compounds with salts: {result_df['has_salt'].sum()}")

    # Test the SDF file writing
    my_sdf_file = "test_compounds.sdf"
    df_to_sdf_file(test_df, my_sdf_file, skip_invalid=False)

    # Test the SDF file reading
    df = sdf_file_to_df(my_sdf_file)
    print(df)
