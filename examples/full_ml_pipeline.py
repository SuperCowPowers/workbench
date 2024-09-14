"""This Script creates a full AWS ML Pipeline with SageWorks

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
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint

# Setup the logger
log = logging.getLogger("sageworks")

if __name__ == "__main__":
    # Create the abalone_data DataSource
    ds = DataSource("s3://sageworks-public-data/common/abalone.csv")

    # Now create a FeatureSet
    ds.to_features("abalone_features")

    # Create the abalone_regression Model
    fs = FeatureSet("abalone_features")
    fs.to_model(
        ModelType.REGRESSOR,
        name="abalone-regression",
        target_column="class_number_of_rings",
        tags=["abalone", "regression"],
        description="Abalone Regression Model",
    )

    # Create the abalone_regression Endpoint
    model = Model("abalone-regression")
    model.to_endpoint(name="abalone-regression-end", tags=["abalone", "regression"])

    # Now we'll run inference on the endpoint
    endpoint = Endpoint("abalone-regression-end")

    # Get a DataFrame of data (not used to train) and run predictions
    athena_table = fs.view("training").table
    df = fs.query(f"SELECT * FROM {athena_table} where training = 0")
    results = endpoint.inference(df)
    print(results[["class_number_of_rings", "prediction"]])
