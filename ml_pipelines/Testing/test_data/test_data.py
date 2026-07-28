"""Create the synthetic 'test_data' Workbench artifacts used by the test suite.

Builds one DataSource of synthetic person data, a FeatureSet, and a regression +
classification model (each with an endpoint). Split out of the old
test_artifacts/create_basic_test_artifacts.py.
"""

import logging
import pandas as pd
from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework, PublicData

log = logging.getLogger("workbench")


# Model features (food is categorical)
FEATURES = ["height", "weight", "age", "iq_score", "likes_dogs", "food"]


def main():
    # DataSource: public synthetic person data with a binned salary_class column
    df = PublicData().get("common/test_data")
    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", utc=True)  # CSV round-trips dates as strings
    bins = [-float("inf"), 130000, 150000, float("inf")]
    df["salary_class"] = pd.cut(df["Salary"], bins=bins, labels=["low", "medium", "high"])
    DataSource(df, name="test_data")

    # FeatureSet
    DataSource("test_data").to_features("test_features", id_column="id", event_time_column="date")

    # Regression model + endpoint
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
    Model("test-regression").to_endpoint(tags=["test", "regression"]).test_inference()

    # Classification model + endpoint
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
    Model("test-classification").to_endpoint(tags=["test", "classification"]).test_inference()


if __name__ == "__main__":
    main()
