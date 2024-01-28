"""This Script creates a simple AWS ML Pipeline with SageWorks

DataSources:
    - abalone_data
FeatureSets:
    - abalone_features
Models:
    - abalone-regression
Endpoints:
    - abalone-regression-end
"""

import sys
from pathlib import Path
from sageworks.core.artifacts.model_core import ModelType
from sageworks.core.transforms.data_loaders.light.csv_to_data_source import CSVToDataSource
from sageworks.core.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint


if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Create the abalone_data DataSource
    my_loader = CSVToDataSource(abalone_data_path, "abalone_data")
    my_loader.set_output_tags("abalone:public")
    my_loader.transform()

    # Create the abalone_features FeatureSet
    data_to_features = DataToFeaturesLight("abalone_data", "abalone_features")
    data_to_features.set_output_tags(["abalone", "public"])
    data_to_features.transform()

    # Create the abalone_regression Model
    features_to_model = FeaturesToModel("abalone_features", "abalone-regression", ModelType.REGRESSOR)
    features_to_model.set_output_tags(["abalone", "regression"])
    features_to_model.transform(target_column="class_number_of_rings", description="Abalone Regression Model")

    # Create the abalone_regression Endpoint
    model_to_endpoint = ModelToEndpoint("abalone-regression", "abalone-regression-end")
    model_to_endpoint.set_output_tags(["abalone", "regression"])
    model_to_endpoint.transform()
