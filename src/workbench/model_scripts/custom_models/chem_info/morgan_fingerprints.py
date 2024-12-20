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

# Local imports
from chem_utils import compute_morgan_fingerprints


# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the training job
# and save the model artifacts to the model directory.
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args = parser.parse_args()

    # This model doesn't get trained, it just a feature creation 'model'

    # Sagemaker seems to get upset if we don't save a model, so we'll create a placeholder model
    placeholder_model = {}
    joblib.dump(placeholder_model, os.path.join(args.model_dir, "model.joblib"))


# Model loading and prediction functions
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


# Input and output functions (basically convert CSV to DataFrame)
def input_fn(input_data, content_type):
    if content_type == "text/csv":
        return pd.read_csv(StringIO(input_data))
    raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    if accept_type == "text/csv":
        return output_df.to_csv(index=False), "text/csv"
    raise RuntimeError(f"{accept_type} not supported!")


# Prediction function
def predict_fn(df, model):

    # Compute the Molecular Fingerprints
    df = compute_morgan_fingerprints(df)
    return df
