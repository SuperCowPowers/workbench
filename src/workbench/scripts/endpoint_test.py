"""
Local test harness for SageMaker model scripts.

Usage:
    python model_script_harness.py <local_script.py> <model_name>

Example:
    python model_script_harness.py pytorch.py aqsol-pytorch-reg

This allows you to test LOCAL changes to a model script against deployed model artifacts.
Evaluation data is automatically pulled from the FeatureSet (training = FALSE rows).

Optional: testing/env.json with additional environment variables
"""

# Force CPU mode BEFORE any PyTorch imports to avoid MPS/CUDA issues on Mac
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_default_device('cpu')
# Disable MPS entirely
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

import sys
import json
import importlib.util
import tempfile
import tarfile
import shutil
import pandas as pd

# Workbench Imports
from workbench.api import Model, FeatureSet

# We'll need boto3 for S3 access
try:
    import boto3
except ImportError:
    print("Error: boto3 is required. Install with: pip install boto3")
    sys.exit(1)


def download_and_extract_model(workbench_model: Model, model_dir: str) -> None:
    """Download model artifact from S3 and extract it."""
    s3_uri = workbench_model.model_data_url()
    print(f"Downloading model from {s3_uri}...")

    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Download to temp file
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        s3.download_file(bucket, key, tmp_path)
        print(f"Downloaded to {tmp_path}")

    # Extract
    print(f"Extracting to {model_dir}...")
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(model_dir)

    # Cleanup temp file
    os.unlink(tmp_path)

    # List contents
    print("Model directory contents:")
    for root, dirs, files in os.walk(model_dir):
        level = root.replace(model_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")


def get_eval_data(workbench_model: Model) -> pd.DataFrame:
    """Get evaluation data from the FeatureSet associated with this model."""
    # Get the FeatureSet
    fs_name = workbench_model.get_input()
    fs = FeatureSet(fs_name)
    if not fs.exists():
        raise ValueError(f"No FeatureSet found: {fs_name}")

    # Get evaluation data (training = FALSE)
    table = workbench_model.training_view().table
    print(f"Querying evaluation data from {table}...")
    eval_df = fs.query(f'SELECT * FROM "{table}" WHERE training = FALSE')
    print(f"Retrieved {len(eval_df)} evaluation rows")

    return eval_df


def load_model_script(script_path: str):
    """Dynamically load the model script module."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("model_script", script_path)
    module = importlib.util.module_from_spec(spec)

    # Add to sys.modules so imports within the script work
    sys.modules["model_script"] = module

    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 3:
        print("Usage: python model_script_harness.py <local_script.py> <model_name>")
        print("\nArguments:")
        print("  local_script.py  - Path to your LOCAL model script to test")
        print("  model_name       - Workbench model name (e.g., aqsol-pytorch-reg)")
        print("\nOptional: testing/env.json with additional environment variables")
        sys.exit(1)

    script_path = sys.argv[1]
    model_name = sys.argv[2]

    # Validate local script exists
    if not os.path.exists(script_path):
        print(f"Error: Local script not found: {script_path}")
        sys.exit(1)

    # Initialize Workbench model
    print(f"Loading Workbench model: {model_name}")
    workbench_model = Model(model_name)
    print(f"Model type: {workbench_model.model_type}")
    print()

    # Create a temporary model directory
    model_dir = tempfile.mkdtemp(prefix="model_harness_")
    print(f"Using model directory: {model_dir}")

    try:
        # Load environment variables from env.json if it exists
        if os.path.exists("testing/env.json"):
            print("Loading environment variables from testing/env.json")
            with open("testing/env.json") as f:
                env_vars = json.load(f)
                for key, value in env_vars.items():
                    os.environ[key] = value
                    print(f"  Set {key} = {value}")
            print()

        # Set up SageMaker environment variables
        os.environ["SM_MODEL_DIR"] = model_dir
        print(f"Set SM_MODEL_DIR = {model_dir}")

        # Download and extract model artifacts
        download_and_extract_model(workbench_model, model_dir)
        print()

        # Load the LOCAL model script
        print(f"Loading LOCAL model script: {script_path}")
        module = load_model_script(script_path)
        print()

        # Check for required functions
        if not hasattr(module, "model_fn"):
            raise AttributeError("Model script must have a model_fn function")
        if not hasattr(module, "predict_fn"):
            raise AttributeError("Model script must have a predict_fn function")

        # Load the model
        print("Calling model_fn...")
        print("-" * 50)
        model = module.model_fn(model_dir)
        print("-" * 50)
        print(f"Model loaded: {type(model)}")
        print()

        # Get evaluation data from FeatureSet
        print("Pulling evaluation data from FeatureSet...")
        df = get_eval_data(workbench_model)
        print(f"Input shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print()

        print("Calling predict_fn...")
        print("-" * 50)
        result = module.predict_fn(df, model)
        print("-" * 50)
        print()

        print("Prediction result:")
        print(f"Output shape: {result.shape}")
        print(f"Output columns: {result.columns.tolist()}")
        print()
        print(result.head(10).to_string())

    finally:
        # Cleanup
        print(f"\nCleaning up model directory: {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
