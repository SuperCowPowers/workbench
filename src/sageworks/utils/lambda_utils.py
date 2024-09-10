"""Helper functions for working with SageWorks Lambda Layer"""

import os
import sys
import shutil
import zipfile
import logging
import subprocess

log = logging.getLogger("sageworks")


def load_lambda_layer():
    # Path to the directory containing zip files in the Lambda layer
    zip_dir = "/opt/python_zipped"
    extract_path = "/tmp/python_unzipped"

    # Create the extraction path if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # Check for available disk space in the extract path
    min_required_space = 1024 * 1024 * 1024  # 1 GB
    total, used, free = shutil.disk_usage(extract_path)
    if free < min_required_space:
        log.error(f"Insufficient disk space. Available: {free / (1024 * 1024):.2f} MB.")
        log.error("Package extraction and installation may fail, if so,  increase ephemeral storage.")
    else:
        log.important(f"Available ephemeral storage should be sufficient: {free / (1024 * 1024):.2f} MB.")

    # Move sklearn from zip_dir to extract_path
    sklearn_dir = os.path.join(zip_dir, "sklearn")
    if os.path.exists(sklearn_dir):
        shutil.move(sklearn_dir, extract_path)
        log.important(f"Moved sklearn from {sklearn_dir} to {extract_path}")

    # Check if the zip directory exists
    if not os.path.exists(zip_dir):
        log.error(f"Zip directory not found: {zip_dir}")
        return

    # Iterate over each file in the zip directory
    for file_name in os.listdir(zip_dir):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(zip_dir, file_name)
            log.important(f"Extracting {zip_path} to {extract_path}")

            # Extract each zip file into the extract path
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

    # Check if xgboost already exists in the extract path
    if os.path.exists(os.path.join(extract_path, "xgboost")):
        log.important("xgboost already exists in the extract path...")

    else:
        # Install xgboost into the extract path
        log.important("Installing xgboost...")
        try:
            subprocess.check_call(["pip", "install", "--target", extract_path, "xgboost"])
            log.important("Successfully installed xgboost...")
        except subprocess.CalledProcessError as e:
            log.critical(f"Failed to install xgboost: {e}")

    # Add the extracted path to the System Path (so imports will find it)
    sys.path.append(extract_path)
