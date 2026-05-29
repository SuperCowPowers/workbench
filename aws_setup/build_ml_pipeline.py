"""This Script creates a basic ML Pipeline for the abalone dataset to test that the Workbench is setup correctly.

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
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.api.data_source import DataSource
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType, ModelFramework
from workbench.api.endpoint import Endpoint

if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "abalone.csv"

    """Check if the AWS Account is Setup Correctly"""
    print("*** AWS Identity Check ***")
    AWSAccountClamp().check_aws_identity()

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
            model_framework=ModelFramework.XGBOOST,
            target_column="class_number_of_rings",
            tags=["abalone", "regression"],
            description="Abalone Regression Model",
        )

    # Create the abalone_regression Endpoint
    if not Endpoint("abalone-regression").exists():
        model = Model("abalone-regression")
        model.to_endpoint(name="abalone-regression", tags=["abalone", "regression"])
