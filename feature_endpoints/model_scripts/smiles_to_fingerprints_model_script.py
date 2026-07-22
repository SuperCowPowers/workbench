# Model: smiles_to_fingerprints_model_script
#
# Description: Computes Morgan count fingerprints (ECFP-style) from SMILES strings.
#     Each of the n_bits positions holds the count of that circular substructure
#     (clamped to uint8), the recommended representation for ADMET property
#     prediction. Salts are handled internally via largest-fragment selection.
#
import argparse
import os
from io import StringIO
import pandas as pd
import json

# Local imports
from molecular_utils.fingerprints import compute_morgan_fingerprints
from fingerprint_config import FP_RADIUS, FP_N_BITS

# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the training job
# and save the model artifacts to the model directory.
#
if __name__ == "__main__":
    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.parse_args()

    # This model doesn't get trained, it just a feature creation 'model'
    # So we don't need to do anything here


# Model loading and prediction functions
def model_fn(model_dir):
    return None


def input_fn(input_data, content_type):
    """Parse input data and return a DataFrame."""
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    # Decode bytes to string if necessary
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))  # Assumes JSON array of records
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    """Supports both CSV and JSON output formats."""
    if "text/csv" in accept_type:
        return output_df.to_csv(index=False), "text/csv"
    elif "application/json" in accept_type:
        return (
            output_df.to_json(orient="records"),
            "application/json",
        )  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


# Prediction function
def predict_fn(df, model):

    # Compute Morgan count fingerprints per the shared config (radius / n_bits).
    # Desalting (largest-fragment selection) is handled inside
    # compute_morgan_fingerprints, so there's no separate standardize step here —
    # this matches how fingerprint models are trained (no train/inference skew).
    df = compute_morgan_fingerprints(df, radius=FP_RADIUS, n_bits=FP_N_BITS)
    return df
