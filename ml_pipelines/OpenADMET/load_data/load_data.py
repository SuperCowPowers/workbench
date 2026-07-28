"""Producer: the shared Open ADMET FeatureSets, one per assay.

Sources both public ExpansionRx wide tables straight from PublicData:

  - training/all_endpoints (5326 rows) -> one FeatureSet per assay
  - testing/all_endpoints  (2282 rows) -> featurized and stashed in the DFStore

Training and test rows are kept strictly apart: the FeatureSets carry *only*
training rows. The test set is featurized here -- once, through the same feature
endpoint the FeatureSets use -- and parked in the DFStore for the per-assay model
scripts to score against, so the same 2282 molecules aren't featurized nine times.

The test set is a distinct set of molecules rather than a subset of the training
table, so it simply has no place in a FeatureSet. Keeping it in the DFStore also
means the per-assay scripts share one consistent featurization -- the same
smiles-to-2d-v1 columns the models were trained on -- which is what keeps the
rx_test_* captures free of train/inference skew.

Both tables get the same log transform (TRANSFORM_CONFIG) so the FeatureSets and
the test frame live in the same space as the model targets.
"""

import numpy as np
import pandas as pd
from workbench.api import Endpoint, FeatureSet, DFStore, PublicData
from workbench.core.transforms.pandas_transforms import PandasToFeatures

# The public wide tables (one row per compound, one column per assay)
TRAIN_DATA = "comp_chem/openadmet/expansionrx/training/all_endpoints"
TEST_DATA = "comp_chem/openadmet/expansionrx/testing/all_endpoints"

# Where the featurized test set lands for the per-assay model scripts.
# Deliberately NOT '/workbench/datasets/open_admet_test_featurized' -- that key holds
# the *blinded* submission set used by support/run_inference.py. Different data.
TEST_STORE_KEY = "/workbench/datasets/open_admet_rx_test_featurized"
TRAIN_STORE_KEY = "/workbench/datasets/open_admet_rx_train_featurized"

# Feature endpoints: 2D descriptors for the model feature space, fingerprints for
# the FeatureSet's compressed column (proximity/neighbor work downstream).
FEATURE_ENDPOINT = "smiles-to-2d-v1"
FINGERPRINT_ENDPOINT = "smiles-to-fingerprints-v0"

# Log transformation config from the OpenADMET tutorial.
# Applied as log10((value + 1) * multiplier); multiplier converts uM -> M.
TRANSFORM_CONFIG = {
    "logd": {"log_transform": False, "multiplier": 1.0},
    "ksol": {"log_transform": True, "multiplier": 1e-6},
    "hlm_clint": {"log_transform": True, "multiplier": 1.0},
    "mlm_clint": {"log_transform": True, "multiplier": 1.0},
    "caco2_papp_a_b": {"log_transform": True, "multiplier": 1e-6},
    "caco2_efflux": {"log_transform": True, "multiplier": 1.0},
    "mppb": {"log_transform": True, "multiplier": 1.0},
    "mbpb": {"log_transform": True, "multiplier": 1.0},
    "mgmb": {"log_transform": True, "multiplier": 1.0},
}

CLASSIFICATION_LABELS = ["low", "moderate", "high"]


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log10 transformations to the assay columns present in df."""
    df = df.copy()
    for col, config in TRANSFORM_CONFIG.items():
        if col in df.columns and config["log_transform"]:
            df[col] = np.log10((df[col] + 1) * config["multiplier"])
    return df


def load_public(data_name: str) -> pd.DataFrame:
    """Pull a public wide table, align the id column, and log-transform the assays."""
    df = PublicData().get(data_name)
    df = df.rename(columns={"id": "molecule_name"})
    df = apply_log_transforms(df)
    print(f"Loaded {len(df)} rows from '{data_name}'")
    return df


def featurize(df: pd.DataFrame, fingerprints: bool = False) -> pd.DataFrame:
    """Append 2D descriptor columns (and optionally fingerprints) via feature endpoints."""
    if fingerprints:
        df = Endpoint(FINGERPRINT_ENDPOINT).inference(df)
    return Endpoint(FEATURE_ENDPOINT).inference(df)


def add_class_column(df: pd.DataFrame, assay: str) -> pd.DataFrame:
    """Add a 3-bin equal-width classification column for the assay."""
    col_min, col_max = df[assay].min(), df[assay].max()
    step = (col_max - col_min) / 3
    bins = [-float("inf"), col_min + step, col_min + 2 * step, float("inf")]
    df["class"] = pd.cut(df[assay], bins=bins, labels=CLASSIFICATION_LABELS).astype(str)
    print(f"  Classification distribution: {df['class'].value_counts().to_dict()}")
    return df


def main():
    df_store = DFStore()
    features = Endpoint(FEATURE_ENDPOINT).output_columns()

    # --- Test set: featurize once, park it in the DFStore, never put it in a FeatureSet ---
    test_df = featurize(load_public(TEST_DATA))
    df_store.upsert(TEST_STORE_KEY, test_df)
    print(f"Stored featurized test set ({len(test_df)} rows) at {TEST_STORE_KEY}")

    # --- Training set: featurize, then split into one FeatureSet per assay ---
    train_df = featurize(load_public(TRAIN_DATA), fingerprints=True)
    df_store.upsert(TRAIN_STORE_KEY, train_df)

    for assay in TRANSFORM_CONFIG:
        fs_name = f"open_admet_{assay}"

        # Rows with a measurement for this assay
        df_assay = train_df.dropna(subset=[assay]).copy()
        df_assay = add_class_column(df_assay, assay)
        df_assay = df_assay[["molecule_name", "smiles", assay, "class"] + features + ["fingerprint"]]

        print(f"Creating FeatureSet: {fs_name} with {len(df_assay)} entries")
        to_features = PandasToFeatures(fs_name)
        to_features.set_input(df_assay, id_column="molecule_name")
        to_features.set_output_tags(["open_admet", assay])
        to_features.transform()

        FeatureSet(fs_name).set_compressed_features(["fingerprint"])


if __name__ == "__main__":
    main()
