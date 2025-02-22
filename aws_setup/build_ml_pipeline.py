"""This Script creates the Workbench Artifacts in AWS needed for the tests

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
from pathlib import Path
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.api.data_source import DataSource
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.api.endpoint import Endpoint


def redis_check():
    """Check if the Redis Database is available"""
    print("*** Redis Database Check ***")
    try:
        from workbench.utils.redis_cache import RedisCache

        RedisCache(prefix="test")
        print("Redis Database Check Success...")
    except RuntimeError as err:
        print(f"Redis Database Check Failed: {err} but this is fine.. Redis is optional")


if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    test_data_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "test_data.csv"
    abalone_data_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "abalone.csv"

    """Check if the AWS Account is Setup Correctly"""
    print("*** AWS Identity Check ***")
    AWSAccountClamp().check_aws_identity()

    # Check that the Redis Database is available
    redis_check()

    # Create the abalone_data DataSource
    if not DataSource("abalone_data").exists():
        DataSource(abalone_data_path, name="abalone_data")

    # Create the abalone_features FeatureSet
    if not FeatureSet("abalone_features").exists():
        ds = DataSource("abalone_data")
        ds.to_features("abalone_features")

    # Create the abalone_regression Model
    if not Model("abalone-regression").exists():
        fs = FeatureSet("abalone_features")
        fs.to_model(
            name="abalone-regression",
            model_type=ModelType.REGRESSOR,
            target_column="class_number_of_rings",
            tags=["abalone", "regression"],
            description="Abalone Regression Model",
        )

    # Create the abalone_regression Endpoint
    if not Endpoint("abalone-regression").exists():
        model = Model("abalone-regression")
        model.to_endpoint(name="abalone-regression", tags=["abalone", "regression"])
