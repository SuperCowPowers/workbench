"""This Script creates the AQSol (Public) Artifacts in AWS/Workbench

DataSources:
    - aqsol_data

FeatureSets:
    - aqsol_features

Models:
    - aqsol-regression
    - aqsol-class

Endpoints:
    - aqsol-regression
    - aqsol-class
"""

import logging
import pandas as pd
import awswrangler as wr

from workbench.api import DataSource, FeatureSet, Model, ModelType, Endpoint

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Get the path to the dataset in S3
    s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the aqsol_data DataSource
    if recreate or not DataSource("aqsol_data").exists():
        # We could create a Datasource directly,  but we're going to add a column to the data
        df = wr.s3.read_csv(s3_path)

        # Create a solubility classification column
        bins = [-float("inf"), -5, -4, float("inf")]
        labels = ["low", "medium", "high"]
        df["solubility_class"] = pd.cut(df["Solubility"], bins=bins, labels=labels)

        # Now we'll create the DataSource with the new column
        DataSource(df, name="aqsol_data")

    # Create the aqsol_features FeatureSet
    if recreate or not FeatureSet("aqsol_features").exists():
        ds = DataSource("aqsol_data")
        ds.to_features("aqsol_features", id_column="id", tags=["aqsol", "public"])

    aqsol_features = [
        "molwt",
        "mollogp",
        "molmr",
        "heavyatomcount",
        "numhacceptors",
        "numhdonors",
        "numheteroatoms",
        "numrotatablebonds",
        "numvalenceelectrons",
        "numaromaticrings",
        "numsaturatedrings",
        "numaliphaticrings",
        "ringcount",
        "tpsa",
        "labuteasa",
        "balabanj",
        "bertzct",
    ]
    # Create the aqsol solubility regression Model
    if recreate or not Model("aqsol-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        m = feature_set.to_model(
            name="aqsol-regression",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=aqsol_features,
            description="AQSol Regression Model",
            tags=["aqsol", "regression"],
        )
        m.set_owner("test")

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-regression").exists():
        m = Model("aqsol-regression")
        end = m.to_endpoint(tags=["aqsol", "regression"])
        end.set_owner("test")
        end.auto_inference()

    # Create the aqsol solubility classification Model
    if recreate or not Model("aqsol-class").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        m = feature_set.to_model(
            name="aqsol-class",
            model_type=ModelType.CLASSIFIER,
            target_column="solubility_class",
            feature_list=aqsol_features,
            description="AQSol Classification Model",
            tags=["aqsol", "classification"],
        )
        m.set_owner("test")
        m.set_class_labels(["low", "medium", "high"])

    # Create the aqsol classification Endpoint
    if recreate or not Endpoint("aqsol-class").exists():
        m = Model("aqsol-class")
        end = m.to_endpoint(tags=["aqsol", "classification"])
        end.set_owner("test")
        end.auto_inference()

    log.info("AQSol Artifacts creation complete.")
