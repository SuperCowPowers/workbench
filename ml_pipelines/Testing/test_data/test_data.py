"""Create the synthetic 'test_data' Workbench artifacts used by the test suite.

Builds one DataSource of synthetic person data, a FeatureSet, and a regression +
classification model (each with an endpoint). Split out of the old
test_artifacts/create_basic_test_artifacts.py.
"""
import logging
import pandas as pd
from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework, Endpoint
from workbench.utils.synthetic_data_generator import SyntheticDataGenerator

log = logging.getLogger("workbench")

# Recreate flag: set True to rebuild artifacts that already exist
RECREATE = False

# Model features (food is categorical)
FEATURES = ["height", "weight", "age", "iq_score", "likes_dogs", "food"]


def main():
    # DataSource: synthetic person data with a binned salary_class column
    if RECREATE or not DataSource("test_data").exists():
        df = SyntheticDataGenerator().person_data()
        bins = [-float("inf"), 130000, 150000, float("inf")]
        df["salary_class"] = pd.cut(df["Salary"], bins=bins, labels=["low", "medium", "high"])
        DataSource(df, name="test_data")

    # FeatureSet
    if RECREATE or not FeatureSet("test_features").exists():
        DataSource("test_data").to_features("test_features", id_column="id", event_time_column="date")

    # Regression model + endpoint
    if RECREATE or not Model("test-regression").exists():
        m = FeatureSet("test_features").to_model(
            name="test-regression",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            feature_list=FEATURES,
            target_column="salary",
            tags=["test", "regression"],
            description="Test Model with Categorical Features",
        )
        m.set_owner("test")
    if RECREATE or not Endpoint("test-regression").exists():
        Model("test-regression").to_endpoint(tags=["test", "regression"]).test_inference()

    # Classification model + endpoint
    if RECREATE or not Model("test-classification").exists():
        m = FeatureSet("test_features").to_model(
            name="test-classification",
            model_type=ModelType.CLASSIFIER,
            model_framework=ModelFramework.XGBOOST,
            feature_list=FEATURES,
            target_column="salary_class",
            tags=["test", "classification"],
            description="Test Classification Model",
        )
        m.set_owner("test")
        m.set_class_labels(["low", "medium", "high"])
    if RECREATE or not Endpoint("test-classification").exists():
        Model("test-classification").to_endpoint(tags=["test", "classification"]).test_inference()


if __name__ == "__main__":
    main()
