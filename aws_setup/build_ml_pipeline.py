"""This Script creates the SageWorks Artifacts in AWS needed for the tests

DataSources:
    - test_data
    - abalone_data
FeatureSets:
    - test_feature_set
    - abalone_feature_set
Models:
    - abalone-regression
Endpoints:
    - abalone-regression-end
"""
import sys
import time
from pathlib import Path
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint
from sageworks.transforms.data_loaders.light.csv_to_data_source import CSVToDataSource
from sageworks.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint


def redis_check():
    """Check if the Redis Database is available"""
    print("*** Redis Database Check ***")
    try:
        from sageworks.utils.redis_cache import RedisCache

        RedisCache(prefix="test")
        print("Redis Database Check Success...")
    except RuntimeError as err:
        print(f"Redis Database Check Failed: {err} but this is fine.. Redis is optional")


if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    test_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.csv"
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    """Check if the AWS Account is Setup Correctly"""
    print("*** AWS Identity Check ***")
    AWSAccountClamp().check_aws_identity()

    # Check that the Redis Database is available
    redis_check()

    # Create the test_data DataSource
    if not DataSource("test_data").exists():
        my_loader = CSVToDataSource(test_data_path, "test_data")
        my_loader.set_output_tags("test:small")
        my_loader.transform()
        print("Waiting for the test_data to be created...")
        time.sleep(10)

    # Create the abalone_data DataSource
    if not DataSource("abalone_data").exists():
        my_loader = CSVToDataSource(abalone_data_path, "abalone_data")
        my_loader.set_output_tags("abalone:public")
        my_loader.transform()
        print("Waiting for the abalone_data to be created...")
        time.sleep(10)

    # Give user a warning about how long the rest of the script will take
    print("\n******************************************************************************")
    print("Note: The rest of this script takes about 20 minutes (AWS finalize/wait times)")
    print("******************************************************************************\n")
    time.sleep(5)

    # Create the test_feature_set FeatureSet
    if not FeatureSet("test_feature_set").exists():
        data_to_features = DataToFeaturesLight("test_data", "test_feature_set")
        data_to_features.set_output_tags(["test", "small"])
        data_to_features.transform(id_column="id", event_time_column="date")

    # Create the abalone_feature_set FeatureSet
    if not FeatureSet("abalone_feature_set").exists():
        data_to_features = DataToFeaturesLight("abalone_data", "abalone_feature_set")
        data_to_features.set_output_tags(["abalone", "public"])
        data_to_features.transform()

    # Create the abalone_regression Model
    if not Model("abalone-regression").exists():
        features_to_model = FeaturesToModel("abalone_feature_set", "abalone-regression")
        features_to_model.set_output_tags(["abalone", "regression"])
        features_to_model.transform(target="class_number_of_rings", description="Abalone Regression Model")

    # Wait for the Model to be created
    while not Model("abalone-regression").exists():
        print("Waiting for the Model to be created...")
        time.sleep(5)

    # Create the abalone_regression Endpoint
    if not Endpoint("abalone-regression-end").exists():
        model_to_endpoint = ModelToEndpoint("abalone-regression", "abalone-regression-end")
        model_to_endpoint.set_output_tags(["abalone", "regression"])
        model_to_endpoint.transform()
