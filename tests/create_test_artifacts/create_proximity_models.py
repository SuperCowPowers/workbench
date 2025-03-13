"""This Script creates a Proximity Model in AWS/Workbench

Models:
    - abalone-prox
    - aqsol-prox

Endpoints:
    - abalone-prox
    - aqsol-prox
"""

import logging

# Workbench Imports
from workbench.api import Model, Endpoint
from workbench.utils.model_utils import proximity_model

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # A Proximity Model based on Abalone Features
    if recreate or not Model("abalone-prox").exists():

        # Create the Proximity Model from our Model
        model = Model("abalone-regression")
        proximity_model(model, "abalone-prox")

    # Create the Proximity Model based on AQSol Features
    if recreate or not Model("aqsol-prox").exists():

        # Create the Proximity Model from our Model
        model = Model("aqsol-regression")
        proximity_model(model, "aqsol-prox")

    # Endpoints for our Proximity Models
    if recreate or not Endpoint("abalone-prox").exists():
        m = Model("abalone-prox")
        m.set_owner("BW")
        end = m.to_endpoint(tags=["proximity", "abalone"])
        end.auto_inference(capture=True)

    if recreate or not Endpoint("aqsol-prox").exists():
        m = Model("aqsol-prox")
        m.set_owner("BW")
        end = m.to_endpoint(tags=["proximity", "aqsol"])
        end.auto_inference(capture=True)
