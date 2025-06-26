"""This Script creates the AQSol Uncertainly Quantification Artifacts in AWS/Workbench

Models:
    - aqsol-uq

Endpoints:
    - aqsol-uq
"""

import logging

from workbench.api import Model, Endpoint

# Get the Workbench logger
log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Get the path to the dataset in S3
    s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = True

    # Check if the Model already exist
    if recreate or not Model("aqsol-uq").exists():

        # Grab our AQSol Regression Model
        model = Model("aqsol-regression")

        # Make a UQ Model
        uq_model = model.uq_model("aqsol-uq")

    # Check if the Endpoint already exists
    if recreate or not Endpoint("aqsol-uq").exists():
        uq_model = Model("aqsol-uq")
        end = uq_model.to_endpoint(name="aqsol-uq", tags=["aqsol", "uq"])

        # Run auto-inference on the Endpoint
        end.auto_inference(capture=True)
