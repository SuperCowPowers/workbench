# Model: tautomerization_processor
#
# Description: The tautomerization_processor model uses RDKit to perform tautomer enumeration
#     and canonicalization of chemical compounds. Tautomerization is the chemical process where
#     compounds can interconvert between structurally distinct forms, often affecting their
#     chemical properties and reactivity. This model provides a robust approach to identifying
#     and processing tautomers, crucial for improving molecular modeling and cheminformatics tasks
#     like virtual screening, QSAR modeling, and property prediction.
#
import argparse
import os
import joblib
from io import StringIO
import pandas as pd
import json

# Local imports
from chem_utils import perform_tautomerization


# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the job and save the model artifacts.
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args = parser.parse_args()

    # This model doesn't get trained; it's a feature processing 'model'

    # Sagemaker expects a model artifact, so we'll save a placeholder
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
    if "text/csv" in accept_type:
        csv_output = output_df.fillna("N/A").to_csv(index=False)  # CSV with N/A for missing values
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


# Prediction function
def predict_fn(df, model):
    # Perform Tautomerization
    df = perform_tautomerization(df)
    return df
