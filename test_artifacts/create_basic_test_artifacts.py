"""This Script creates the Workbench Artifacts in AWS needed for the tests

DataSources:
    - test_data
    - abalone_data
FeatureSets:
    - test_features
    - abalone_features
Models:
    - test-regression
    - test-classification
    - abalone-regression
Endpoints:
    - test-regression
    - test-classification
    - abalone-regression-end
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from workbench.api import DataSource, FeatureSet, Model, ModelType, Endpoint
from workbench.utils.test_data_generator import TestDataGenerator

# Setup the logger
log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "abalone.csv"
    karate_graph = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "karate_graph.json"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the test_data DataSource
    if recreate or not DataSource("test_data").exists():
        # Create a new Data Source from a dataframe of test data
        test_data = TestDataGenerator()
        df = test_data.person_data()

        # Create classification column
        bins = [-float("inf"), 130000, 150000, float("inf")]
        labels = ["low", "medium", "high"]
        df["salary_class"] = pd.cut(df["Salary"], bins=bins, labels=labels)
        DataSource(df, name="test_data")

    # Create the test_features FeatureSet
    if recreate or not FeatureSet("test_features").exists():
        ds = DataSource("test_data")
        ds.to_features("test_features", id_column="id", event_time_column="date")

    # Create the Test Model (with categorical features)
    features = ["height", "weight", "age", "iq_score", "likes_dogs", "food"]  # Food is categorical
    if recreate or not Model("test-regression").exists():
        fs = FeatureSet("test_features")
        m = fs.to_model(
            name="test-regression",
            model_type=ModelType.REGRESSOR,
            feature_list=features,
            target_column="salary",
            tags=["test", "regression"],
            description="Test Model with Categorical Features",
        )
        m.set_owner("test")

    # Create the Test Endpoint
    if recreate or not Endpoint("test-regression").exists():
        model = Model("test-regression")
        end = model.to_endpoint(tags=["test", "regression"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Test Data Classification Model
    if recreate or not Model("test-classification").exists():
        fs = FeatureSet("test_features")
        m = fs.to_model(
            name="test-classification",
            model_type=ModelType.CLASSIFIER,
            feature_list=features,
            target_column="salary_class",
            tags=["test", "classification"],
            description="Test Classification Model",
        )
        m.set_owner("test")
        m.set_class_labels(["low", "medium", "high"])

    # Create the Test Endpoint
    if recreate or not Endpoint("test-classification").exists():
        model = Model("test-classification")
        end = model.to_endpoint(tags=["test", "classification"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create the abalone_data DataSource
    if recreate or not DataSource("abalone_data").exists():
        DataSource(abalone_data_path, name="abalone_data")

    # Create the abalone_features FeatureSet
    if recreate or not FeatureSet("abalone_features").exists():
        ds = DataSource("abalone_data")
        ds.to_features("abalone_features")

    # Create the abalone_regression Model
    if recreate or not Model("abalone-regression").exists():
        fs = FeatureSet("abalone_features")
        features = [
            "length",
            "diameter",
            "height",
            "whole_weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight",
            "sex",
        ]
        m = fs.to_model(
            name="abalone-regression",
            model_type=ModelType.REGRESSOR,
            feature_list=features,
            target_column="class_number_of_rings",
            tags=["abalone", "regression"],
            description="Abalone Regression Model",
        )
        m.set_owner("test")

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression").exists():
        model = Model("abalone-regression")
        end = model.to_endpoint(name="abalone-regression", tags=["abalone", "regression"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
