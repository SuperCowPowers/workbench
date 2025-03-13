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
from workbench.api import FeatureSet, Model, ModelType, Endpoint
from workbench.utils.model_utils import get_custom_script_path

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # A Proximity Model based on Abalone Features
    if recreate or not Model("abalone-prox").exists():
        script_path = get_custom_script_path("proximity", "feature_space_proximity.template")

        # Get Feature and Target Columns from the existing Abalone Model
        m = Model("abalone-regression")
        features = m.features()
        target = m.target()

        # Create the Proximity Model from our FeatureSet
        fs = FeatureSet("abalone_features")
        fs.to_model(
            name="abalone-prox",
            model_type=ModelType.TRANSFORMER,
            feature_list=features,
            target_column=target,
            description="Proximity Model for Abalone Features",
            tags=["proximity", "abalone"],
            custom_script=script_path,
        )

    # Create the Proximity Model based on AQSol Features
    if recreate or not Model("aqsol-prox").exists():
        script_path = get_custom_script_path("proximity", "feature_space_proximity.template")

        # Get Feature and Target Columns from the existing AQSol Model
        m = Model("aqsol-knn-reg")
        features = m.features()
        target = m.target()

        # Create the Proximity Model from our FeatureSet
        fs = FeatureSet("aqsol_features")
        fs.to_model(
            name="aqsol-prox",
            model_type=ModelType.TRANSFORMER,
            feature_list=features,
            target_column=target,
            description="Proximity Model for AQSol Features",
            tags=["proximity", "aqsol"],
            custom_script=script_path,
        )

    # Endpoints for our Proximity Models
    if recreate or not Endpoint("abalone-prox").exists():
        m = Model("abalone-prox")
        m.set_owner("BW")
        end = m.to_endpoint(tags=["proximity", "abalone"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    if recreate or not Endpoint("aqsol-prox").exists():
        m = Model("aqsol-prox")
        m.set_owner("BW")
        end = m.to_endpoint(tags=["proximity", "aqsol"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
