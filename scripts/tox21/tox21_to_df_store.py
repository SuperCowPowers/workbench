import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Workbench Imports
from workbench.api.df_store import DFStore

# Import the new mol_tagging module
from workbench.utils.chem_utils.mol_tagging import tag_molecules, filter_by_tags, get_tag_summary
from workbench.utils.chem_utils.fingerprints import compute_morgan_fingerprints
from workbench.utils.chem_utils.projections import project_fingerprints


def prep_sdf_file(filepath: str) -> pd.DataFrame:
    """
    Prepare SDF file for analysis using the new tagging system.
    """

    # Load SDF file directly into a DataFrame
    df = PandasTools.LoadSDF(filepath, smilesName="smiles", molColName="molecule", includeFingerprints=False)
    print(f"Loaded {len(df)} compounds from {filepath}")
    print(f"Columns: {df.columns.tolist()}")

    # Standardize column names
    df.rename(columns={"ID": "id"}, inplace=True)

    # Convert Formula weight to numeric
    df["FW"] = pd.to_numeric(df["FW"], errors="coerce")

    # Full list of assay columns for toxicity detection
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

    # Convert assay columns to numeric
    df[assay_cols] = df[assay_cols].apply(pd.to_numeric, errors="coerce")

    # Create toxic_any column (1 if any assay is positive)
    df["toxic_any"] = (df[assay_cols] == 1).any(axis=1).astype(int)
    print(f"Toxicity distribution:\n{df['toxic_any'].value_counts(dropna=False)}")

    # Ensure we have SMILES column (might already be there from LoadSDF)
    if "smiles" not in df.columns and "molecule" in df.columns:
        df["smiles"] = df["molecule"].apply(lambda mol: Chem.MolToSmiles(mol) if mol else None)

    # Apply the new tagging system
    # Use all tag categories by default
    df = tag_molecules(
        df,
        smiles_column="smiles",
        tag_column="tags",
        tag_categories=None,  # This will include all: metals, halogens, druglike, structure
    )

    # Add toxicity-specific tags
    # Add "tox21" tag for toxic compounds
    df["tags"] = df.apply(lambda row: row["tags"] + ["tox21_toxic"] if row["toxic_any"] == 1 else row["tags"], axis=1)

    # Add "tox21_clean" tag for non-toxic compounds with data
    df["tags"] = df.apply(
        lambda row: (
            row["tags"] + ["tox21_clean"]
            if row["toxic_any"] == 0 and not df.loc[row.name, assay_cols].isna().all()
            else row["tags"]
        ),
        axis=1,
    )

    # Add specific assay tags for positive results
    for assay in assay_cols:
        assay_tag = f"positive_{assay.lower().replace('-', '_')}"
        df["tags"] = df.apply(lambda row: row["tags"] + [assay_tag] if row[assay] == 1 else row["tags"], axis=1)

    # Print tag summary
    print("\nTag Summary:")
    tag_summary = get_tag_summary(df)
    print(tag_summary.head(15))

    # Compute Fingerprints (assuming these functions exist somewhere)
    df = compute_morgan_fingerprints(df, radius=2)
    df = project_fingerprints(df, projection="UMAP")

    # Drop the molecule column to save space
    if "molecule" in df.columns:
        df.drop(columns=["molecule"], inplace=True)

    # Print final summary
    print(f"\nFinal DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Show examples of different molecule types
    print("\n=== Example molecules by category ===")

    # Drug-like molecules
    druglike = filter_by_tags(df, require=["ro5_pass"])
    print(f"Drug-like molecules (Ro5 pass): {len(druglike)}")

    # Clean, drug-like, non-toxic molecules
    ideal = filter_by_tags(df, require=["ro5_pass", "tox21_clean"], exclude=["heavy_metal", "highly_halogenated"])
    print(f"Ideal molecules (drug-like, non-toxic, no metals/heavy halogenation): {len(ideal)}")

    # Problematic molecules
    problematic = filter_by_tags(df, require=["tox21_toxic"])
    print(f"Toxic molecules: {len(problematic)}")

    return df


def analyze_toxicity_patterns(df: pd.DataFrame) -> None:
    """
    Analyze patterns in toxicity data using the tag system.
    """
    print("\n=== Toxicity Pattern Analysis ===")

    # Check correlation between structural features and toxicity
    toxic_df = filter_by_tags(df, require=["tox21_toxic"])
    clean_df = filter_by_tags(df, require=["tox21_clean"])

    print(f"\nTotal toxic compounds: {len(toxic_df)}")
    print(f"Total clean compounds: {len(clean_df)}")

    # Analyze tag distribution in toxic vs clean
    if len(toxic_df) > 0:
        print("\nTop tags in toxic compounds:")
        toxic_tags = get_tag_summary(toxic_df)
        # Remove the tox21 tags themselves from analysis
        toxic_tags = toxic_tags[~toxic_tags.index.str.startswith("tox21_")]
        toxic_tags = toxic_tags[~toxic_tags.index.str.startswith("positive_")]
        print(toxic_tags.head(10))

    if len(clean_df) > 0:
        print("\nTop tags in clean compounds:")
        clean_tags = get_tag_summary(clean_df)
        clean_tags = clean_tags[~clean_tags.index.str.startswith("tox21_")]
        clean_tags = clean_tags[~clean_tags.index.str.startswith("positive_")]
        print(clean_tags.head(10))

    # Check specific patterns
    print("\n=== Specific Pattern Analysis ===")

    # Heavy metals and toxicity
    metal_toxic = filter_by_tags(toxic_df, require=["heavy_metal"])
    print(f"Toxic compounds with heavy metals: {len(metal_toxic)} ({len(metal_toxic) / len(toxic_df) * 100:.1f}%)")

    # Halogenation and toxicity
    halogen_toxic = filter_by_tags(toxic_df, require=["highly_halogenated"])
    print(f"Toxic compounds highly halogenated: {len(halogen_toxic)} ({len(halogen_toxic) / len(toxic_df) * 100:.1f}%)")

    # Drug-likeness in toxic compounds
    druglike_toxic = filter_by_tags(toxic_df, require=["ro5_pass"])
    print(f"Toxic compounds passing Ro5: {len(druglike_toxic)} ({len(druglike_toxic) / len(toxic_df) * 100:.1f}%)")


if __name__ == "__main__":
    # Load the training and test sets
    training = "/Users/briford/data/workbench/tox21/training/tox21_10k_data_all.sdf"
    test = "/Users/briford/data/workbench/tox21/testing/tox21_10k_challenge_test.sdf"
    final_test = "/Users/briford/data/workbench/tox21/testing/tox21_10k_challenge_score.sdf"

    # Process the training data
    print("Processing training data...")
    df = prep_sdf_file(training)

    # Analyze toxicity patterns
    analyze_toxicity_patterns(df)

    # Store in DFStore
    df_store = DFStore()
    df_store.upsert("/datasets/chem_info/tox21_training", df)
    print(f"\nStored training data to /datasets/chem_info/tox21_training")

    # Optionally process test data
    # print("\nProcessing test data...")
    # test_df = prep_sdf_file(test)
    # df_store.upsert("/datasets/chem_info/tox21_test", test_df)
