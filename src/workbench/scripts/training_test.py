"""
Local test harness for SageMaker training scripts.

Usage:
    python training_test.py <model_script.py> <featureset_name>

Example:
    python training_test.py ../model_scripts/pytorch_model/generated_model_script.py caco2-class-features
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from workbench.api import FeatureSet


def get_training_data(featureset_name: str) -> pd.DataFrame:
    """Get training data from the FeatureSet."""
    fs = FeatureSet(featureset_name)
    return fs.pull_dataframe()


def main():
    if len(sys.argv) < 3:
        print("Usage: python training_test.py <model_script.py> <featureset_name>")
        sys.exit(1)

    script_path = sys.argv[1]
    featureset_name = sys.argv[2]

    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    # Create temp directories
    model_dir = tempfile.mkdtemp(prefix="training_model_")
    train_dir = tempfile.mkdtemp(prefix="training_data_")
    output_dir = tempfile.mkdtemp(prefix="training_output_")

    print(f"Model dir: {model_dir}")
    print(f"Train dir: {train_dir}")

    try:
        # Get training data and save to CSV
        print(f"Loading FeatureSet: {featureset_name}")
        df = get_training_data(featureset_name)
        print(f"Data shape: {df.shape}")

        train_file = os.path.join(train_dir, "training_data.csv")
        df.to_csv(train_file, index=False)

        # Set up environment
        env = os.environ.copy()
        env["SM_MODEL_DIR"] = model_dir
        env["SM_CHANNEL_TRAIN"] = train_dir
        env["SM_OUTPUT_DATA_DIR"] = output_dir

        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")

        # Run the script
        cmd = [sys.executable, script_path, "--model-dir", model_dir, "--train", train_dir]
        result = subprocess.run(cmd, env=env)

        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("Training completed successfully!")
        else:
            print(f"Training failed with return code: {result.returncode}")
        print("=" * 60)

    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
