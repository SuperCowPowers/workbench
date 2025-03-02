# Model: morgan_fingerprints
#
# Description: The morgan_fingerprints model uses RDKit to generate molecular fingerprints,
#     specifically focusing on Morgan (Extended Connectivity Fingerprints, ECFP) representations.
#     Morgan fingerprints are circular fingerprints widely used in cheminformatics to capture
#     the local chemical environment around atoms. They are highly effective for tasks like
#     similarity searches, virtual screening, and quantitative structure-activity relationship (QSAR) modeling,
#     providing a robust and modern approach to encoding molecular structure.
#
import argparse
import os
import joblib
from io import StringIO
import pandas as pd
import json

# Local imports
from local_utils import compute_morgan_fingerprints


# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the training job
# and save the model artifacts to the model directory.
#
if __name__ == "__main__":
    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    args = parser.parse_args()

    # This model doesn't get trained, it just a feature creation 'model'

    # Sagemaker seems to get upset if we don't save a model, so we'll create a placeholder model
    placeholder_model = {}
    joblib.dump(placeholder_model, os.path.join(args.model_dir, "model.joblib"))


# Model loading and prediction functions
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


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
        return output_df.to_json(orient="records"), "application/json"  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


# Prediction function
def predict_fn(df, model):

    # Compute the Molecular Fingerprints
    df = compute_morgan_fingerprints(df)
    return df
