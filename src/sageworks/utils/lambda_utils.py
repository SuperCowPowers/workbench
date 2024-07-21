"""Helper functions for working with SageWorks Labmda Layer"""

import os
import sys
import shutil
import zipfile
import logging

log = logging.getLogger("sageworks")


def load_lambda_layer():
    # Path to the directory containing zip files in the Lambda layer
    zip_dir = "/opt/python_zipped"
    extract_path = "/tmp/python_unzipped"

    # Create the extraction path if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # Move sklearn from zip_dir to extract_path
    sklearn_dir = os.path.join(zip_dir, "sklearn")
    if os.path.exists(sklearn_dir):
        shutil.move(sklearn_dir, extract_path)
        print(f"Moved sklearn from {sklearn_dir} to {extract_path}")

    # Iterate over each file in the zip directory
    for file_name in os.listdir(zip_dir):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(zip_dir, file_name)
            print(f"Extracting {zip_path} to {extract_path}")

            # Extract each zip file into the extract path
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

    # Add the extracted path to the System Path (so imports will find it)
    sys.path.append(extract_path)
