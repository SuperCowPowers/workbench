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
    - abalone-regression-val  (20% held-out validation set)
Endpoints:
    - test-regression
    - test-classification
    - abalone-regression-end
    - abalone-regression-val
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework, Endpoint
from workbench.utils.synthetic_data_generator import SyntheticDataGenerator

# Setup the logger
log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the test_data DataSource
    if recreate or not DataSource("test_data").exists():
        # Create a new Data Source from a dataframe of test data
        test_data = SyntheticDataGenerator()
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
            model_framework=ModelFramework.XGBOOST,
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
        end.test_inference()

    # Test Data Classification Model
    if recreate or not Model("test-classification").exists():
        fs = FeatureSet("test_features")
        m = fs.to_model(
            name="test-classification",
            model_type=ModelType.CLASSIFIER,
            model_framework=ModelFramework.XGBOOST,
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
        end.test_inference()

    # Create the abalone_data DataSource
    if recreate or not DataSource("abalone_data").exists():
        DataSource(abalone_data_path, name="abalone_data")

    # Create the abalone_features FeatureSet
    if recreate or not FeatureSet("abalone_features").exists():
        ds = DataSource("abalone_data")
        ds.to_features("abalone_features")

    # Shared by both abalone models below (defined outside the guards so it's
    # available whichever model needs creating)
    abalone_feature_list = [
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "sex",
    ]

    # Create the abalone_regression Model
    if recreate or not Model("abalone-regression").exists():
        fs = FeatureSet("abalone_features")
        features = abalone_feature_list
        m = fs.to_model(
            name="abalone-regression",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
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
        end.test_inference()

    # Create an abalone Model with a designated validation set. Holding out 20% of the
    # rows gives this model a populated val_predictions.csv alongside oof_predictions.csv,
    # so both training captures have a fixture to exercise.
    if recreate or not Model("abalone-regression-val").exists():
        fs = FeatureSet("abalone_features")
        all_ids = fs.pull_dataframe()[fs.id_column]
        validation_ids = all_ids.sample(frac=0.2, random_state=42).tolist()
        log.important(f"Holding out {len(validation_ids)} of {len(all_ids)} rows as the validation set")
        m = fs.to_model(
            name="abalone-regression-val",
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            feature_list=abalone_feature_list,
            target_column="class_number_of_rings",
            tags=["abalone", "regression", "validation"],
            description="Abalone Regression Model (20% held-out validation set)",
            validation_ids=validation_ids,
        )
        m.set_owner("test")

    # Create the abalone-regression-val Endpoint
    if recreate or not Endpoint("abalone-regression-val").exists():
        model = Model("abalone-regression-val")
        end = model.to_endpoint(name="abalone-regression-val", tags=["abalone", "regression", "validation"])

        # test_inference populates a capture; cross_fold_inference reads oof_predictions.csv
        end.test_inference()
        end.cross_fold_inference()
