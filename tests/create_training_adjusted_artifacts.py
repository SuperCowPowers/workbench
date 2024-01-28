"""This Script creates the 'Training Adjusted' Artifacts in AWS needed for the tests

FeatureSets:
    - Create a training view for abalone_features
Models:
    - abalone-regression-100
Endpoints:
    - abalone-regression-end-100
"""

import sys
import time
import logging
from pathlib import Path
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.artifacts.model_core import ModelCore, ModelType
from sageworks.core.artifacts.endpoint_core import EndpointCore
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

# Setup the logger
log = logging.getLogger("sageworks")


if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create a training view of the test_features
    log.important("Creating training view for abalone_features...")
    fs = FeatureSetCore("abalone_features")
    fs.create_training_view("id", hold_out_ids=range(100))  # Just the first 100 ids

    # Create the abalone_regression Model
    if recreate or not ModelCore("abalone-regression-100").exists():
        features_to_model = FeaturesToModel(
            "abalone_features", "abalone-regression-100", model_type=ModelType.REGRESSOR
        )
        features_to_model.set_output_tags(["abalone", "regression"])
        features_to_model.transform(
            target_column="class_number_of_rings", description="Abalone Regression Model", train_all_data=True
        )
        log.info("Waiting for the Model to be created...")
        time.sleep(10)

    # Create the abalone_regression Endpoint
    if recreate or not EndpointCore("abalone-regression-end-100").exists():
        model_to_endpoint = ModelToEndpoint("abalone-regression-100", "abalone-regression-end-100")
        model_to_endpoint.set_output_tags(["abalone", "regression"])
        model_to_endpoint.transform()
