#!/usr/bin/env python3
import os
import sys
import json
import tarfile
import subprocess
import logging
import boto3
from urllib.parse import urlparse

# Set up logging
logger = logging.getLogger('sagemaker-entry-point')
logger.setLevel(logging.INFO)


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
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_path
            ])
            logger.info("Requirements installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {e}")
            sys.exit(1)
    else:
        logger.info(f"No requirements file found at {requirements_path}")


def setup_environment():
    """Set up SageMaker environment variables."""
    env_vars = {
        "SM_MODEL_DIR": "/opt/ml/model",
        "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
        "SM_CHANNEL_TRAIN": "/opt/ml/input/data/train",
        "SM_OUTPUT_DIR": "/opt/ml/output",
        "SM_INPUT_DIR": "/opt/ml/input",
        "SM_INPUT_CONFIG_DIR": "/opt/ml/input/config"
    }

    for key, value in env_vars.items():
        os.environ[key] = str(value)
        os.makedirs(value, exist_ok=True)

    logger.info(f"SageMaker environment initialized.")


def main():
    logger.info("Starting Workbench training container...")

    # Debug available environment variables
    logger.info("Available environment variables:")
    for key in os.environ:
        logger.info(f"  {key}: {os.environ[key]}")

    # Recursively list out all files in /opt/ml
    logger.info("Contents of /opt/ml:")
    for root, dirs, files in os.walk("/opt/ml"):
        for file in files:
            logger.info(f"  {root}/{file}")

    # Load hyperparameters
    hyperparams_path = '/opt/ml/input/config/hyperparameters.json'
    if not os.path.exists(hyperparams_path):
        logger.error("hyperparameters.json not found!")
        sys.exit(1)

    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
    logger.info(f"Hyperparameters: {hyperparams}")

    # Get program name from hyperparameters
    if 'sagemaker_program' in hyperparams:
        training_script = hyperparams['sagemaker_program'].strip('"\'')
    else:
        logger.error("sagemaker_program not found in hyperparameters!")
        sys.exit(1)

    logger.info(f"Using training_script: {training_script}")

    # Get source directory from hyperparameters
    if 'sagemaker_submit_directory' in hyperparams:
        code_directory = hyperparams['sagemaker_submit_directory'].strip('"\'')

        # Handle S3 vs local path
        if code_directory.startswith('s3://'):
            code_directory = download_and_extract_s3(code_directory)
        elif not os.path.exists(code_directory):
            logger.error(f"Local code directory not found: {code_directory}")
            sys.exit(1)

    # Install requirements if present
    install_requirements(os.path.join(code_directory, "requirements.txt"))

    # Set up environment variables
    setup_environment()

    # Find training script (entry point)
    entry_point = os.path.join(code_directory, training_script)
    if not os.path.exists(entry_point):
        logger.error(f"Entry point not found: {entry_point}")
        sys.exit(1)

    logger.info(f"Executing: {entry_point}")

    # Execute the training script with SageMaker arguments
    cmd = [
        sys.executable, entry_point,
        "--model-dir", os.environ["SM_MODEL_DIR"],
        "--output-data-dir", os.environ["SM_OUTPUT_DATA_DIR"],
        "--train", os.environ["SM_CHANNEL_TRAIN"]
    ]

    try:
        os.execv(sys.executable, cmd)
    except Exception as e:
        logger.error(f"Failed to execute entry point: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
