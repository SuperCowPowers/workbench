"""This Script creates the AQSol Uncertainly Quantification Artifacts in AWS/Workbench

Models:
    - aqsol-uq
    - aqsol-uq-100

Endpoints:
    - aqsol-uq
    - aqsol-uq-100
"""

import logging
import argparse

from workbench.api import Model, Endpoint

# Get the Workbench logger
log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Take a recreate flag from the command line
    parser = argparse.ArgumentParser(description="Create AQSol UQ Artifacts")
    parser.add_argument("--recreate", action="store_true", help="Recreate the artifacts if they already exist")
    args = parser.parse_args()
    recreate = args.recreate

    # Get the path to the dataset in S3
    s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"

    # Check if the Model already exist
    if recreate or not Model("aqsol-uq").exists():

        # Grab our AQSol Regression Model
        model = Model("aqsol-regression")

        # Make a UQ Model
        uq_model = model.uq_model("aqsol-uq", train_all_data=True)

    # Check if the Endpoint already exists
    if recreate or not Endpoint("aqsol-uq").exists():
        uq_model = Model("aqsol-uq")
        end = uq_model.to_endpoint(tags=["aqsol", "uq"])

        # Run auto-inference on the Endpoint
        end.auto_inference(capture=True)
        end.cross_fold_inference()
