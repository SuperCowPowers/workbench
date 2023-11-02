"""This Script creates the 'Training Adjusted' Artifacts in AWS needed for the tests

FeatureSets:
    - Create a training view for abalone_feature_set
Models:
    - abalone-regression-100
Endpoints:
    - abalone-regression-end-100
"""
import sys
import time
from pathlib import Path
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model, ModelType
from sageworks.artifacts.endpoints.endpoint import Endpoint
from sageworks.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint


if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create a training view of the test_feature_set
    print("Creating training view for abalone_feature_set...")
    fs = FeatureSet("abalone_feature_set")
    fs.create_training_view("id", hold_out_ids=range(100))  # Just the first 100 ids

    # Create the abalone_regression Model
    if recreate or not Model("abalone-regression-100").exists():
        features_to_model = FeaturesToModel(
            "abalone_feature_set", "abalone-regression-100", model_type=ModelType.REGRESSOR
        )
        features_to_model.set_output_tags(["abalone", "regression"])
        features_to_model.transform(
            target_column="class_number_of_rings", description="Abalone Regression Model", train_all_data=True
        )
        print("Waiting for the Model to be created...")
        time.sleep(10)

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression-end-100").exists():
        model_to_endpoint = ModelToEndpoint("abalone-regression-100", "abalone-regression-end-100")
        model_to_endpoint.set_output_tags(["abalone", "regression"])
        model_to_endpoint.transform()
