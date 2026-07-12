"""Create the 'abalone' Workbench artifacts used by the test suite.

Loads the repo's abalone.csv into a DataSource, builds a FeatureSet, and a regression
model + endpoint. Split out of the old test_artifacts/create_basic_test_artifacts.py.
"""

import sys
import logging
from pathlib import Path
from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework, Endpoint

log = logging.getLogger("workbench")

# Recreate flag: set True to rebuild artifacts that already exist
RECREATE = False

# abalone.csv ships in the repo's data/ directory (alongside the workbench package)
ABALONE_CSV = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "abalone.csv"

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
    if RECREATE or not DataSource("abalone_data").exists():
        DataSource(ABALONE_CSV, name="abalone_data")

    # FeatureSet
    if RECREATE or not FeatureSet("abalone_features").exists():
        DataSource("abalone_data").to_features("abalone_features")

    # Regression model + endpoint
    if RECREATE or not Model("abalone-regression").exists():
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
    if RECREATE or not Endpoint("abalone-regression").exists():
        Model("abalone-regression").to_endpoint(
            name="abalone-regression", tags=["abalone", "regression"]
        ).test_inference()


if __name__ == "__main__":
    main()
