"""
Local test harness for SageMaker model scripts.

Usage:
    python model_script_harness.py <local_script.py> <model_name>

Example:
    python model_script_harness.py pytorch.py aqsol-reg-pytorch

This allows you to test LOCAL changes to a model script against deployed model artifacts.
Evaluation data is automatically pulled from the FeatureSet (training = FALSE rows).

Optional: testing/env.json with additional environment variables
"""

import os
import sys
import json
import importlib.util
import tempfile
import shutil
import pandas as pd
import torch

# Workbench Imports
from workbench.api import Model, FeatureSet
from workbench.utils.pytorch_utils import download_and_extract_model

# Force CPU mode BEFORE any PyTorch imports to avoid MPS/CUDA issues on Mac
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_default_device("cpu")
# Disable MPS entirely
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False


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
        print("  model_name       - Workbench model name (e.g., aqsol-reg-pytorch)")
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
    print(f"Model Framework: {workbench_model.model_framework}")
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
        s3_uri = workbench_model.model_data_url()
        download_and_extract_model(s3_uri, model_dir)
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
