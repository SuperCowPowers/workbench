#!/usr/bin/env python3
import os
import sys
import shutil
import json
import tarfile
import subprocess
import logging
import boto3
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_extract_s3(s3_uri, target_dir="/opt/ml/code"):
    """Download and extract code package from S3."""
    logger.info(f"Downloading source package from {s3_uri}...")
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_tar = "/tmp/code_package.tar.gz"

    try:
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_tar)
        logger.info(f"Download successful: {os.path.getsize(local_tar)} bytes")

        os.makedirs(target_dir, exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=target_dir, numeric_owner=True)
        return target_dir
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        sys.exit(1)


def install_requirements(requirements_path):
    """Install Python dependencies from requirements file."""
    if os.path.exists(requirements_path):
        logger.info(f"Installing dependencies from {requirements_path}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            logger.info("Requirements installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {e}")
            sys.exit(1)
    else:
        logger.info(f"No requirements file found at {requirements_path}")


def include_code_and_meta_for_inference(model_dir, code_dir, entry_point):
    """Include code and some metadata for the inference container"""
    logger.info("Including code and metadata for inference...")

    # Create inference metadata file
    inference_metadata = {"inference_script": entry_point}

    # Write metadata to model directory
    metadata_path = os.path.join(model_dir, "inference-metadata.json")
    with open(metadata_path, "w") as fp:
        json.dump(inference_metadata, fp)

    # Copy code to model directory, copy ALL files and directories recursively (except __pycache__)
    # Also list all files/directories that are being copied
    for item in os.listdir(code_dir):
        if item == "__pycache__":
            continue
        src, dst = os.path.join(code_dir, item), os.path.join(model_dir, item)
        shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy2(src, dst)
        logger.info(f"Copied: {src} -> {dst}")


def main():
    logger.info("Starting Workbench training container...")

    # Load hyperparameters
    hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
    if not os.path.exists(hyperparams_path):
        logger.error("hyperparameters.json not found!")
        sys.exit(1)

    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)
    logger.info(f"Hyperparameters: {hyperparams}")

    # Get program name from hyperparameters
    if "sagemaker_program" in hyperparams:
        training_script = hyperparams["sagemaker_program"].strip("\"'")
    else:
        logger.error("sagemaker_program not found in hyperparameters!")
        sys.exit(1)

    logger.info(f"Using training_script: {training_script}")

    # Get source directory from hyperparameters
    if "sagemaker_submit_directory" in hyperparams:
        code_directory = hyperparams["sagemaker_submit_directory"].strip("\"'")

        # Handle S3 vs local path
        if code_directory.startswith("s3://"):
            code_directory = download_and_extract_s3(code_directory)
        elif not os.path.exists(code_directory):
            logger.error(f"Local code directory not found: {code_directory}")
            sys.exit(1)

    # Install requirements if present
    install_requirements(os.path.join(code_directory, "requirements.txt"))

    # Find training script
    training_script_path = os.path.join(code_directory, training_script)
    if not os.path.exists(training_script_path):
        logger.error(f"Training script not found: {training_script_path}")
        sys.exit(1)

    logger.info(f"Executing: {training_script_path}")

    # Add the code directory to the Python path
    os.environ["PYTHONPATH"] = f"{code_directory}:{os.environ.get('PYTHONPATH', '')}"

    # Call the training script and then include code and meta for inference
    try:
        subprocess.check_call(
            [
                sys.executable,
                training_script_path,
                "--model-dir",
                os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                "--output-data-dir",
                os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
                "--train",
                os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
            ]
        )

        # After training completes, include code and meta in the model.tar.gz
        include_code_and_meta_for_inference(
            model_dir=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
            code_dir=code_directory,
            entry_point=training_script,
        )

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute training script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
