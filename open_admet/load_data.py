"""Load the Open ADMET data using Workbench API"""

import numpy as np
import pandas as pd
from workbench.api import DataSource, FeatureSet, Endpoint, ParameterStore, DFStore
from workbench.core.transforms.pandas_transforms import PandasToFeatures

# Log transformation config from OpenADMET tutorial
# Uses lowercase column names (matching what RDKit endpoint outputs)
TRANSFORM_CONFIG = {
    "logd": {"log_transform": False, "multiplier": 1.0},
    "ksol": {"log_transform": True, "multiplier": 1e-6},
    "hlm_clint": {"log_transform": True, "multiplier": 1.0},
    "mlm_clint": {"log_transform": True, "multiplier": 1.0},
    "caco_2_papp_a_b": {"log_transform": True, "multiplier": 1e-6},
    "caco_2_efflux": {"log_transform": True, "multiplier": 1.0},
    "mppb": {"log_transform": True, "multiplier": 1.0},
    "mbpb": {"log_transform": True, "multiplier": 1.0},
    "mgmb": {"log_transform": True, "multiplier": 1.0},
}


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log10 transformations to assay columns based on OpenADMET tutorial."""
    df_transformed = df.copy()

    for col, config in TRANSFORM_CONFIG.items():
        if col not in df_transformed.columns:
            continue

        if config["log_transform"]:
            # Apply: log10((value + 1) * multiplier)
            df_transformed[col] = np.log10((df_transformed[col] + 1) * config["multiplier"])

    return df_transformed


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, replace spaces and special chars with underscores."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace(">", "_")
    return df


def main():
    # Access the Parameter Store and DataFrame Store
    params = ParameterStore()
    df_store = DFStore()

    # Load the original training data
    """
    df = pd.read_csv("train_data.csv")
    print(f"Loaded {len(df)} rows from train_data.csv")

    # Normalize column names first (lowercase, underscores)
    df = normalize_columns(df)
    print(f"Normalized column names: {list(df.columns)}")

    # Apply log transformations to assay columns
    print("Applying log transformations...")
    df_transformed = apply_log_transforms(df)
    print(f"Transformed columns: {list(TRANSFORM_CONFIG.keys())}")

    # Save transformed data to CSV for DataSource creation
    df_transformed.to_csv("train_data_xformed.csv", index=False)
    print("Saved transformed data to train_data_xformed.csv")

    # Create a new DataSource with transformed data
    # ds = DataSource("train_data_xformed.csv", name="open_admet_xformed")
    # df = ds.pull_dataframe()
    """

    # We've already created the DataSource "open_admet_xformed" in Workbench
    df = DataSource("open_admet_xformed").pull_dataframe()
    print(f"Pulled {len(df)} rows from DataSource 'open_admet_xformed'")

    # Run the data through our Fingerprint Endpoint
    fp_end = Endpoint("smiles-to-fingerprints-v0")
    df_features = fp_end.inference(df)

    # Run the data through our RDKit+Mordred Feature Endpoint
    rdkit_end = Endpoint("smiles-to-taut-md-stereo-v1")
    df_features = rdkit_end.inference(df_features)

    # Shove this into the DFStore for inspection/use later
    df_store.upsert("/workbench/datasets/open_admet_xformed_featurized", df_features)

    # Grab the Feature List created by the Endpoint
    features = params.get("/workbench/feature_lists/rdkit_mordred_stereo_v1")

    # Now Split these into separate FeatureSets for each assay
    for assay in TRANSFORM_CONFIG.keys():
        fs_name = f"open_admet_{assay}"

        # Pull all rows with non-null values for this assay
        df_assay = df_features.dropna(subset=[assay])

        # Just keep the molecule_name, smiles, assay, and feature columns
        keep_columns = ["molecule_name", "smiles", assay] + features + ["fingerprint"]
        df_assay = df_assay[keep_columns]

        # Create a Feature Set
        print(f"Creating FeatureSet: {fs_name} with {len(df_assay)} entries")
        to_features = PandasToFeatures(fs_name)
        to_features.set_input(df_assay, id_column="molecule_name")
        to_features.set_output_tags(["open_admet", assay])
        to_features.transform()

        # Set our compressed features for this FeatureSet
        fs = FeatureSet(fs_name)
        fs.set_compressed_features(["fingerprint"])


if __name__ == "__main__":
    main()
