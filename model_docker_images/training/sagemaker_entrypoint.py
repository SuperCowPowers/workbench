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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sagemaker-entry-point')


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
            tar.extractall(path=target_dir)

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
    logger.info("Starting SageMaker container entry point")

    # Load hyperparameters
    hyperparams_path = '/opt/ml/input/config/hyperparameters.json'
    if not os.path.exists(hyperparams_path):
        logger.error("hyperparameters.json not found!")
        sys.exit(1)

    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)

    # Get program name from hyperparameters or environment
    if 'sagemaker_program' in hyperparams:
        program = hyperparams['sagemaker_program'].strip('"\'')
        os.environ['SAGEMAKER_PROGRAM'] = program
    elif 'SAGEMAKER_PROGRAM' in os.environ:
        program = os.environ['SAGEMAKER_PROGRAM']
    else:
        logger.error("sagemaker_program not found in hyperparameters or environment!")
        sys.exit(1)

    logger.info(f"Using program: {program}")

    # Get source directory
    submit_dir = "/opt/ml/code"
    if 'sagemaker_submit_directory' in hyperparams:
        submit_dir_value = hyperparams['sagemaker_submit_directory'].strip('"\'')

        # Handle S3 vs local path
        if submit_dir_value.startswith('s3://'):
            submit_dir = download_and_extract_s3(submit_dir_value)
        else:
            submit_dir = submit_dir_value
            if not os.path.exists(submit_dir):
                logger.error(f"Local directory not found: {submit_dir}")
                sys.exit(1)

    # Install requirements if present
    install_requirements(os.path.join(submit_dir, "requirements.txt"))

    # Set up environment variables
    setup_environment()

    # Find entry point script
    entry_point = os.path.join(submit_dir, program)
    if not os.path.exists(entry_point):
        logger.error(f"Entry point not found: {entry_point}")
        sys.exit(1)

    logger.info(f"Executing: {program}")

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
