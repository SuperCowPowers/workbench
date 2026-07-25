"""Create the 'abalone' Workbench artifacts used by the test suite.

Loads the public abalone dataset into a DataSource, builds a FeatureSet, and a
regression model + endpoint. Split out of the old test_artifacts/create_basic_test_artifacts.py.
"""

import logging
from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework, PublicData

log = logging.getLogger("workbench")


FEATURES = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
    "sex",
]


def main():
    # DataSource
    DataSource(PublicData().get("testing/abalone"), name="abalone_data")

    # FeatureSet
    DataSource("abalone_data").to_features("abalone_features", id_column="auto")

    # Regression model + endpoint
    m = FeatureSet("abalone_features").to_model(
        name="abalone-regression",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        feature_list=FEATURES,
        target_column="class_number_of_rings",
        tags=["abalone", "regression"],
        description="Abalone Regression Model",
    )
    m.set_owner("test")
    Model("abalone-regression").to_endpoint(name="abalone-regression", tags=["abalone", "regression"]).test_inference()


if __name__ == "__main__":
    main()
