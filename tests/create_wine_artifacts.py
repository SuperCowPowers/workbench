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
import time

from pathlib import Path
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint

from sageworks.transforms.data_loaders.light.csv_to_data_source import CSVToDataSource
from sageworks.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint

if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    wine_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "wine_dataset.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the abalone_data DataSource
    if recreate or not DataSource("wine_data").exists():
        my_loader = CSVToDataSource(wine_data_path, "wine_data")
        my_loader.set_output_tags("wine:classification")
        my_loader.transform()
        print("Waiting for the wine_data to be created...")
        time.sleep(5)

    # Create the wine_features FeatureSet
    if recreate or not FeatureSet("wine_features").exists():
        data_to_features = DataToFeaturesLight("wine_data", "wine_features")
        data_to_features.set_output_tags(["wine", "classification"])
        data_to_features.transform(target="wine_class", description="Wine Classification Features")

    # Create the wine classification Model
    if recreate or not Model("wine-classification").exists():
        features_to_model = FeaturesToModel("wine_features", "wine-classification")
        features_to_model.set_output_tags(["wine", "classification"])
        features_to_model.transform(
            target="wine_class", description="Wine Classification Model", model_type="classifier"
        )

    # Create the wine classification Endpoint
    if recreate or not Endpoint("wine-classification-end").exists():
        model_to_endpoint = ModelToEndpoint("wine-classification", "wine-classification-end")
        model_to_endpoint.set_output_tags(["wine", "classification"])
        model_to_endpoint.transform()
