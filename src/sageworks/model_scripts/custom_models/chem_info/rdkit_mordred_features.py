# Welcome to the custom script template
#
# This script is designed to be used with the SageMaker Scikit-learn container.
# It runs a simple training and inference job using a Scikit-learn model.
# This script will work 'as is' just to get you started, but you should modify it
# to suit your own use case.
#
# We've placed a few comments throughout the script to help guide you.
# Each section with have a 'REPLACE:' in the comment to indicate where you should
# replace the code with your own.
#
# For Regression Models, basically remove all the LabelEncoder code.
#
# Also review the requirements.txt file to ensure you have all the necessary libraries.
#
# If you have any questions, please don't hesitate to touch base with SageWorks support.
#
import argparse
import os
import joblib
from io import StringIO
import pandas as pd

# Local imports
from chem_utils import compute_molecular_descriptors


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

    # Compute the Molecular Descriptors
    df = compute_molecular_descriptors(df)
    return df
