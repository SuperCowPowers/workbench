"""This Script creates a full AWS ML Pipeline with Workbench

DataSource:
    - abalone_data
FeatureSet:
    - abalone_features
Model:
    - abalone-regression
Endpoint:
    - abalone-regression-end
"""

import logging
from workbench.api.data_source import DataSource
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.api.endpoint import Endpoint

# Setup the logger
log = logging.getLogger("workbench")

if __name__ == "__main__":
    # Create the abalone_data DataSource
    ds = DataSource("s3://workbench-public-data/common/abalone.csv")

    # Now create a FeatureSet
    ds.to_features("abalone_features")

    # Create the abalone_regression Model
    fs = FeatureSet("abalone_features")
    fs.to_model(
        name="abalone-regression",
        model_type=ModelType.REGRESSOR,
        target_column="class_number_of_rings",
        tags=["abalone", "regression"],
        description="Abalone Regression Model",
    )

    # Create the abalone_regression Endpoint
    model = Model("abalone-regression")
    model.to_endpoint(name="abalone-regression", tags=["abalone", "regression"])

    # Now we'll run inference on the endpoint
    endpoint = Endpoint("abalone-regression")

    # Get a DataFrame of data (not used to train) and run predictions
    athena_table = fs.view("training").table
    df = fs.query(f"SELECT * FROM {athena_table} where training = FALSE")
    results = endpoint.inference(df)
    print(results[["class_number_of_rings", "prediction"]])
