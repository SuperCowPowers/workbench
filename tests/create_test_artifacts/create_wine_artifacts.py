"""This Script creates the Classification Artifacts in AWS/SageWorks

DataSources:
    - wine_data
FeatureSets:
    - wine_feature_set
Models:
    - wine-classification
Endpoints:
    - wine-classification-end
"""

import sys

from pathlib import Path
from sageworks.api import DataSource, FeatureSet, Model, ModelType, Endpoint


if __name__ == "__main__":

    # Get the path to the dataset in the repository data directory
    wine_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "wine_dataset.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the wine_data DataSource
    if recreate or not DataSource("wine_data").exists():
        DataSource(wine_data_path, name="wine_data")

    # Create the wine_features FeatureSet
    if recreate or not FeatureSet("wine_features").exists():
        ds = DataSource("wine_data")
        ds.to_features("wine_features", id_column="auto", tags=["wine", "classification"])

    # Create the wine classification Model
    if recreate or not Model("wine-classification").exists():
        fs = FeatureSet("wine_features")
        m = fs.to_model(
            model_type=ModelType.CLASSIFIER,
            name="wine-classification",
            target_column="wine_class",
            tags=["wine", "classification"],
            description="Wine Classification Model",
        )
        m.set_owner("test")

    # Create the wine classification Endpoint
    if recreate or not Endpoint("wine-classification-end").exists():
        m = Model("wine-classification")
        end = m.to_endpoint("wine-classification-end", tags=["wine", "classification"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
