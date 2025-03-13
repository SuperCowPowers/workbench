"""This Script creates a Proximity Model in AWS/Workbench

Models:
    - abalone-prox

Endpoints:
    - abalone-prox
"""

import importlib.resources
import logging
from pathlib import Path

from workbench.api import FeatureSet, Model, ModelType, Endpoint


log = logging.getLogger("workbench")


# We have some custom model script in "workbench.model_scripts.custom_models"
def get_custom_script_path(script_name: str) -> Path:
    with importlib.resources.path("workbench.model_scripts.custom_models.proximity", script_name) as script_path:
        return script_path


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = True

    # A Proximity Model based on Abalone Features
    if recreate or not Model("abalone-prox").exists():
        script_path = get_custom_script_path("feature_space_proximity.template")

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

    # Endpoints for our Proximity Model
    if recreate or not Endpoint("abalone-prox").exists():
        m = Model("abalone-prox")
        m.set_owner("BW")
        end = m.to_endpoint(tags=["proximity", "abalone"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
