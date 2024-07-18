"""This Script creates the SageWorks Artifacts in AWS needed for the tests

DataSources:
    - test_data
    - abalone_data
FeatureSets:
    - test_features
    - abalone_features
Models:
    - abalone-regression
Endpoints:
    - abalone-regression-end
"""

import sys
import logging
from pathlib import Path
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint

from sageworks.utils.test_data_generator import TestDataGenerator
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

# Setup the logger
log = logging.getLogger("sageworks")

if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWS Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"
    karate_graph = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "karate_graph.json"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the test_data DataSource
    if recreate or not DataSource("test_data").exists():
        # Create a new Data Source from a dataframe of test data
        test_data = TestDataGenerator()
        df = test_data.person_data()
        DataSource(df, name="test_data")

    # Create the abalone_data DataSource
    if recreate or not DataSource("abalone_data").exists():
        DataSource(abalone_data_path, name="abalone_data")

    # Create the test_features FeatureSet
    if recreate or not FeatureSet("test_features").exists():
        ds = DataSource("test_data")
        ds.to_features("test_features", id_column="id", event_time_column="date")

    # Create the abalone_features FeatureSet
    if recreate or not FeatureSet("abalone_features").exists():
        ds = DataSource("abalone_data")
        ds.to_features("abalone_features")

    # Create the abalone_regression Model
    if recreate or not Model("abalone-regression").exists():
        fs = FeatureSet("abalone_features")
        m = fs.to_model(
            ModelType.REGRESSOR,
            name="abalone-regression",
            target_column="class_number_of_rings",
            tags=["abalone", "regression"],
            description="Abalone Regression Model",
        )
        m.set_owner("test")

    # Create the abalone_regression Model
    if recreate or not Model("abalone-regression-full").exists():
        fs = FeatureSet("abalone_features")
        m = fs.to_model(
            ModelType.REGRESSOR,
            name="abalone-regression-full",
            target_column="class_number_of_rings",
            tags=["abalone", "regression"],
            description="Abalone Regression Model",
            train_all_data=True,
        )
        m.set_owner("test")

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression-end").exists():
        model = Model("abalone-regression")
        end = model.to_endpoint(name="abalone-regression-end", tags=["abalone", "regression"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
