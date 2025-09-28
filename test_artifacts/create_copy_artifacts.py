"""This Script creates the Classification Artifacts in AWS/Workbench

DataSources:
    - wine_data_copy
FeatureSets:
    - wine_feature_set_copy
Models:
    - wine-classification-copy
Endpoints:
    - wine-classification-copy
"""

import sys

from pathlib import Path
from workbench.api import DataSource, FeatureSet, Model, ModelType, Endpoint


if __name__ == "__main__":

    # Get the path to the dataset in the repository data directory
    wine_data_copy_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "wine_dataset.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the wine_data_copy DataSource
    if recreate or not DataSource("wine_data_copy").exists():
        DataSource(wine_data_copy_path, name="wine_data_copy")

    # Create the wine_features_copy FeatureSet
    if recreate or not FeatureSet("wine_features_copy").exists():
        ds = DataSource("wine_data_copy")
        ds.to_features("wine_features_copy", tags=["wine", "classification"])

    # Create the wine classification Model
    if recreate or not Model("wine-classification-copy").exists():
        fs = FeatureSet("wine_features_copy")
        m = fs.to_model(
            name="wine-classification-copy",
            model_type=ModelType.CLASSIFIER,
            target_column="wine_class",
            tags=["wine", "classification"],
            description="Wine Classification Model",
        )
        m.set_owner("test")

    # Create the wine classification Endpoint
    if recreate or not Endpoint("wine-classification-copy").exists():
        m = Model("wine-classification-copy")
        end = m.to_endpoint("wine-classification-copy", tags=["wine", "classification"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
