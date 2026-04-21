# Model: smiles_to_3d_full_model_script
#
# Description: Computes Boltzmann-weighted 3D conformer-based molecular descriptors
#     from SMILES strings. Uses adaptive conformer counts (50-300 based on
#     rotatable bonds) and Boltzmann-weighted ensemble averaging for higher
#     quality descriptors at the cost of longer per-molecule compute time.
#
#     Total: 74 3D descriptors (same features as the fast v1 endpoint)
#
#     Designed for async SageMaker endpoints with batch_size=1.
#     Per-molecule compute time: seconds (rigid) to minutes (flexible).
#
import argparse
import os
from io import StringIO
import logging
import pandas as pd
import json

# Local imports
from molecular_utils.mol_standardize import standardize
from molecular_utils.mol_descriptors_3d import compute_descriptors_3d

SCRIPT_VERSION = "0.1.0"

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
    args = parser.parse_args()

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
    use_explicit_na = False
    if "text/csv" in accept_type:
        if use_explicit_na:
            csv_output = output_df.fillna("N/A").to_csv(index=False)  # CSV with N/A for missing values
        else:
            csv_output = output_df.to_csv(index=False)
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return (
            output_df.to_json(orient="records"),
            "application/json",
        )  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


# Prediction function
def predict_fn(df, model):
    logger = logging.getLogger("workbench")
    logger.info(f"smiles_to_3d_full_model_script v{SCRIPT_VERSION} — processing {len(df)} molecules")

    # Standardize the molecule (extract salts) first
    df = standardize(df, extract_salts=True)

    # Compute Boltzmann-weighted 3D descriptors
    df = compute_descriptors_3d(df, mode="full")

    return df
