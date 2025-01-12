import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Workbench Imports
from workbench.api.df_store import DFStore
from workbench.utils.chem_utils import add_compound_tags, compute_morgan_fingerprints, project_fingerprints


def prep_sdf_file(filepath: str) -> pd.DataFrame:

    # Load SDF file directly into a DataFrame
    df = PandasTools.LoadSDF(filepath, smilesName="smiles", molColName="molecule", includeFingerprints=False)
    print(df.head())
    print(f"Loaded {len(df)} compounds from {filepath}")
    print(f"Columns: {df.columns}")
    df.rename(columns={"ID": "id"}, inplace=True)

    # Column Information
    """
    | `Formula`         | Chemical formula of the compound.                                                             |
    | `FW`              | Molecular weight (Formula Weight) of the compound.                                            |
    | `DSSTox_CID`      | Unique identifier from the Distributed Structure-Searchable Toxicity (DSSTox) database.       |                                               |
    | `ID`              | Internal identifier for the compound.                                                         |
    | `smiles`          | Simplified Molecular Input Line Entry System representation of the compound's structure.      |
    | `molecule`        | Molecular structure information (format may vary). 
    | `NR-AR`           | Nuclear receptor assay for androgen receptor.     
    | `SR-HSE`          | Stress response assay for heat shock element.  |
    | `SR-ARE`          | Stress response assay for antioxidant response element.                                       |
    | `NR-Aromatase`    | Nuclear receptor assay for aromatase enzyme.                                                  |
    | `NR-ER-LBD`       | Nuclear receptor assay for estrogen receptor ligand-binding domain.                           |
    | `NR-AhR`          | Nuclear receptor assay for aryl hydrocarbon receptor.                                         |
    | `SR-MMP`          | Stress response assay for mitochondrial membrane potential.                                   |
    | `NR-ER`           | Nuclear receptor assay for estrogen receptor.                                                 |
    | `NR-PPAR-gamma`   | Nuclear receptor assay for peroxisome proliferator-activated receptor gamma.                  |
    | `SR-p53`          | Stress response assay for p53 tumor suppressor protein.                                       |
    | `SR-ATAD5`        | Stress response assay for ATAD5 (ATPase family AAA domain-containing protein 5).              |
    | `NR-AR-LBD`       | Nuclear receptor assay for androgen receptor ligand-binding domain.                           |
    """

    # Convert Formula weight to numeric
    df["FW"] = pd.to_numeric(df["FW"], errors="coerce")

    # Full list of assay columns for toxicity detection (0/1/NaN) for values
    assay_cols = [
        "NR-AR",  # Nuclear Receptor - Androgen Receptor
        "SR-HSE",  # Stress Response - Heat Shock Element
        "SR-ARE",  # Stress Response - Antioxidant Response Element
        "NR-Aromatase",  # Nuclear Receptor - Aromatase
        "NR-ER-LBD",  # Nuclear Receptor - Estrogen Receptor (Ligand Binding Domain)
        "NR-AhR",  # Nuclear Receptor - Aryl Hydrocarbon Receptor
        "SR-MMP",  # Stress Response - Mitochondrial Membrane Potential
        "NR-ER",  # Nuclear Receptor - Estrogen Receptor
        "NR-PPAR-gamma",  # Nuclear Receptor - Peroxisome Proliferator-Activated Receptor Gamma
        "SR-p53",  # Stress Response - p53 Tumor Suppressor Protein
        "SR-ATAD5",  # Stress Response - ATPase Family AAA Domain-Containing Protein 5
        "NR-AR-LBD",  # Nuclear Receptor - Androgen Receptor (Ligand Binding Domain)
    ]

    # Convert to numeric, coercing errors to NaN
    df[assay_cols] = df[assay_cols].apply(pd.to_numeric, errors="coerce")

    # Set toxic_any to 1 if any of the toxicity columns has a 1
    df["toxic_any"] = (df[assay_cols] == 1).any(axis=1).astype(int)
    print(df["toxic_any"].value_counts(dropna=False))

    # Convert SMILES to RDKit molecule objects (vectorized)
    if "molecule" not in df.columns:
        df["molecule"] = df["smiles"].apply(Chem.MolFromSmiles)

    # Add Compound Tags
    df = add_compound_tags(df, mol_column="molecule")

    # Now we have a 'tag' column with a list of tags in it, let's add "tox21" to the tags if the compound is toxic
    df["tags"] = df.apply(lambda row: row["tags"] + ["tox21"] if row["toxic_any"] == 1 else row["tags"], axis=1)

    # Compute Fingerprints
    df = compute_morgan_fingerprints(df, radius=2)

    # Project Fingerprints to 2D space
    df = project_fingerprints(df, projection="UMAP")

    # df = project_fingerprints(df, projection="TSNE")
    # df.rename(columns={"x": "x_tsne", "y": "y_tsne"}, inplace=True)
    # Convert compound tags to string and check for substrings, then convert to 0/1
    # df["toxic_tag"] = df["tags"].astype(str).str.contains("toxic").astype(int)
    # df["druglike_tag"] = df["tags"].astype(str).str.contains("druglike").astype(int)

    # Drop the molecule column
    df.drop(columns=["molecule"], inplace=True)

    # Return the prepared DataFrame
    return df


if __name__ == "__main__":

    # Load the training and test sets
    training = "/Users/briford/data/workbench/tox21/training/tox21_10k_data_all.sdf"
    test = "/Users/briford/data/workbench/tox21/testing/tox21_10k_challenge_test.sdf"
    final_test = "/Users/briford/data/workbench/tox21/testing/tox21_10k_challenge_score.sdf"

    # Process the data and put it into the DFStore
    df_store = DFStore()
    df = prep_sdf_file(training)
    df_store.upsert("/datasets/chem_info/tox21", df)

    # Okay, so the test datasets have a different format than the training dataset (so skip for now)
    """
    df = prep_sdf_file(test)
    df_store.upsert("/datasets/chem_info/tox21_test", df)
    df = prep_sdf_file(final_test)
    df_store.upsert("/datasets/chem_info/tox21_final_test", df)
    """
