"""This Script creates the Workbench Local Artifacts needed for local development tests

These live on the local filesystem under WORKBENCH_LOCAL_PATH (default
~/.workbench/local) and require no AWS. The names are specific to local testing so
they never collide with the AWS test artifacts in ../test_artifacts.

LocalDataSources:
    - local_test_data
LocalFeatureSets:
    - local_test_features
LocalModels:
    - local-test-regression      (20 held-out validation ids, weights on 5 rows)
LocalEndpoints:
    - local-test-regression
"""

import logging

import numpy as np
import pandas as pd

from workbench.local import LocalDataSource, LocalFeatureSet, LocalModel, LocalEndpoint
from workbench.local import ModelType, ModelFramework

# Setup the logger
log = logging.getLogger("workbench")

# Data is generated here rather than with SyntheticDataGenerator, which imports
# workbench.api and so requires AWS config. This script runs with none.

# The model's row roles, kept here so the publish script can assert they survive
VALIDATION_IDS = list(range(180, 200))
SAMPLE_WEIGHTS = {0: 2.5, 1: 2.5, 2: 0.5, 3: 0.5, 4: 1.5}
FEATURES = ["feature_0", "feature_1", "feature_2", "feature_3"]


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the local_test_data LocalDataSource
    if recreate or not LocalDataSource("local_test_data").exists():
        rng = np.random.default_rng(42)
        rows = 200
        df = pd.DataFrame({f"feature_{i}": rng.normal(0, 1, rows) for i in range(4)})
        df["target"] = 2 * df["feature_0"] - df["feature_1"] + 0.5 * df["feature_2"] + rng.normal(0, 0.2, rows)
        df.insert(0, "id", range(rows))
        LocalDataSource(df, name="local_test_data")

    # Create the local_test_features LocalFeatureSet
    if recreate or not LocalFeatureSet("local_test_features").exists():
        ds = LocalDataSource("local_test_data")
        ds.to_features("local_test_features", id_column="id")

    # Create the Local Regression Model
    # Note: all three row roles are exercised so publish() has something to replay
    if recreate or LocalModel("local-test-regression").training_state().get("state") != "completed":
        fs = LocalFeatureSet("local_test_features")
        fs.to_model(
            name="local-test-regression",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="target",
            feature_list=FEATURES,
            validation_ids=VALIDATION_IDS,
            sample_weights=SAMPLE_WEIGHTS,
        )

    # Create the Local Endpoint
    if recreate or not LocalEndpoint("local-test-regression").exists():
        model = LocalModel("local-test-regression")
        model.to_endpoint()

    # Run inference so the endpoint has a capture to compare against later
    endpoint = LocalEndpoint("local-test-regression")
    if recreate or not endpoint.list_captures():
        eval_df = LocalFeatureSet("local_test_features").pull_dataframe()
        endpoint.inference(eval_df, capture_name="local_holdout")

    # Report what we have
    model = LocalModel("local-test-regression")
    log.important(f"Local model state: {model.training_state().get('state')}")
    log.important(f"Out-of-fold predictions: {len(model.oof_predictions())} rows")
    log.important(f"Validation predictions: {len(model.validation_predictions())} rows")
    log.important(f"Endpoint captures: {endpoint.list_captures()}")
