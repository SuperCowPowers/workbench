"""This Script creates realtime AQSol (Public) Artifacts in AWS/Workbench


Endpoints:
    - tautomerize-v0-rt
    - tautomerize-v0-rt-fast1
    - smiles-to-md-v0-rt
    - aqsol-mol-class-rt
"""

import logging

# Workbench Imports
from workbench.api import Model, Endpoint


log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the realtime Tautomerize Endpoint
    if recreate or not Endpoint("tautomerize-v0-rt").exists():
        m = Model("tautomerize-v0")
        m.set_owner("BW")
        end = m.to_endpoint(name="tautomerize-v0-rt", tags=["smiles", "tautomerization", "realtime"], serverless=False)

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create realtime endpoint for testing with a ml.c7i.large instance
    if recreate or not Endpoint("tautomerize-v0-rt-fast1").exists():
        m = Model("tautomerize-v0")
        m.set_owner("BW")
        end = m.to_endpoint(
            name="tautomerize-v0-rt-fast1",
            tags=["smiles", "tautomerization", "realtime"],
            serverless=False,
            instance="ml.c7i.large",
        )

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create realtime endpoint for testing with a ml.c7i.xlarge instance
    if recreate or not Endpoint("tautomerize-v0-rt-fast2").exists():
        m = Model("tautomerize-v0")
        m.set_owner("BW")
        end = m.to_endpoint(
            name="tautomerize-v0-rt-fast2",
            tags=["smiles", "tautomerization", "realtime"],
            serverless=False,
            instance="ml.c7i.xlarge",
        )

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create a realtime endpoint for Molecular Descriptors Transformer Model
    if recreate or not Endpoint("smiles-to-md-v0-rt").exists():
        m = Model("smiles-to-md-v0")
        m.set_owner("BW")
        end = m.to_endpoint(
            name="smiles-to-md-v0-rt", tags=["smiles", "molecular descriptors", "realtime"], serverless=False
        )

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create a realtime endpoint for AQSOL solubility classification model
    if recreate or not Endpoint("aqsol-mol-class-rt").exists():
        m = Model("aqsol-mol-class")
        m.set_owner("BW")
        end = m.to_endpoint(name="aqsol-mol-class-rt", tags=["aqsol", "class", "realtime"], serverless=False)
