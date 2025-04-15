# Model: Meta Endpoint Example
# This script is a template for creating a custom meta endpoint in AWS Workbench.
from io import StringIO
import pandas as pd
import json

# Workbench Bridges imports
try:
    from workbench_bridges.endpoints.fast_inference import fast_inference
except ImportError:
    print("workbench_bridges not found, this is fine for training...")


# Not Used: We need to define this function for SageMaker
def model_fn(model_dir):
    return None


def input_fn(input_data, content_type):
    """Parse input data and return a DataFrame."""
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    # Decode bytes to string if necessary
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    # Support CSV and JSON input formats
    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))  # Assumes JSON array of records
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    """Supports both CSV and JSON output formats."""
    if "text/csv" in accept_type:
        csv_output = output_df.to_csv(index=False)
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


# Prediction function
def predict_fn(df, model):

    # Call inference on an endpoint
    df = fast_inference("abalone-regression", df)
    return df
