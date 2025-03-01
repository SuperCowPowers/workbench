#!/usr/bin/env python
import os
import sys
import json
import tarfile
import subprocess
import logging
import boto3
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        logger.info(f"Download successful, tar file size: {os.path.getsize(local_tar)} bytes")

        os.makedirs(target_dir, exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=target_dir)

        logger.info(f"Files in {target_dir} after extraction: {os.listdir(target_dir)}")
        return target_dir
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        sys.exit(1)


def install_requirements(requirements_path):
    """Install Python dependencies from requirements file."""
    if os.path.exists(requirements_path):
        logger.info(f"Installing dependencies from {requirements_path}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_path
            ])
            logger.info("Requirements installation completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {str(e)}")
            sys.exit(1)
    else:
        logger.info(f"No requirements file found at {requirements_path}")


def setup_sagemaker_environment():
    """Set up SageMaker environment variables based on /opt/ml structure."""
    env_vars = {
        "SM_MODEL_DIR": "/opt/ml/model",
        "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
        "SM_CHANNEL_TRAIN": "/opt/ml/input/data/train",
        "SM_OUTPUT_DIR": "/opt/ml/output",
        "SM_INPUT_DIR": "/opt/ml/input",
        "SM_INPUT_CONFIG_DIR": "/opt/ml/input/config"
    }

    # Set the environment variables
    for key, value in env_vars.items():
        os.environ[key] = str(value)

    logger.info(f"Set SageMaker environment variables: {list(env_vars.keys())}")


def main():
    logger.info("Starting SageMaker container entry point")

    # Read hyperparameters
    hyperparameters_path = '/opt/ml/input/config/hyperparameters.json'
    if not os.path.exists(hyperparameters_path):
        logger.error("Error: hyperparameters.json not found!")
        sys.exit(1)

    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
        logger.info(f"Hyperparameters: {hyperparameters}")

    # Set up environment based on hyperparameters
    # Get program name from hyperparameters or environment variable
    if 'sagemaker_program' in hyperparameters:
        program = hyperparameters['sagemaker_program'].strip('"\'')
        os.environ['SAGEMAKER_PROGRAM'] = program
    elif 'SAGEMAKER_PROGRAM' in os.environ:
        program = os.environ['SAGEMAKER_PROGRAM']
    else:
        logger.error("Error: sagemaker_program not found in hyperparameters or environment!")
        sys.exit(1)

    logger.info(f"Using program: {program}")

    # Get source directory from hyperparameters
    if 'sagemaker_submit_directory' in hyperparameters:
        submit_dir_value = hyperparameters['sagemaker_submit_directory'].strip('"\'')
        logger.info(f"Source directory: {submit_dir_value}")

        # Check if it's an S3 URI or a local path
        if submit_dir_value.startswith('s3://'):
            logger.info(f"Downloading source from S3: {submit_dir_value}")
            submit_dir = download_and_extract_s3(submit_dir_value)
        else:
            logger.info(f"Using local source directory: {submit_dir_value}")
            submit_dir = submit_dir_value
            # Verify the directory exists
            if not os.path.exists(submit_dir):
                logger.error(f"Local directory not found: {submit_dir}")
                sys.exit(1)

        # Install requirements
        install_requirements(os.path.join(submit_dir, "requirements.txt"))
    else:
        logger.info("No sagemaker_submit_directory specified, assuming code is already in /opt/ml/code")
        submit_dir = "/opt/ml/code"

        # Check if directory exists
        if not os.path.exists(submit_dir):
            logger.error(f"Code directory {submit_dir} does not exist!")
            sys.exit(1)

        # List code directory contents for debugging
        logger.info(f"Contents of {submit_dir}:")
        try:
            output = subprocess.check_output(['ls', '-la', submit_dir])
            logger.info(output.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to list directory: {e}")

    # Set up SageMaker environment variables
    setup_sagemaker_environment()

    # Ensure directories exist
    os.makedirs(os.environ["SM_MODEL_DIR"], exist_ok=True)
    os.makedirs(os.environ["SM_OUTPUT_DATA_DIR"], exist_ok=True)

    # Locate entry point script
    entry_point = os.path.join(submit_dir, program)
    if not os.path.exists(entry_point):
        logger.error(f"Error: Entry point '{entry_point}' not found!")
        sys.exit(1)

    logger.info(f"Running entry point: {entry_point}")
    sys.stdout.flush()

    # Execute with proper arguments
    cmd = [
        sys.executable, entry_point,
        "--model-dir", os.environ["SM_MODEL_DIR"],
        "--output-data-dir", os.environ["SM_OUTPUT_DATA_DIR"],
        "--train", os.environ["SM_CHANNEL_TRAIN"]
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    # Replace current process with the entry point script and arguments
    try:
        os.execv(sys.executable, cmd)
    except Exception as e:
        logger.error(f"Failed to execute entry point: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
