"""This Script creates some 100 percent trained Models in AWS/Workbench

Models:
    - abalone-regression-100
    - aqsol-regression-100
Endpoints:
    - abalone-regression-100
    - aqsol-regression-100
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

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the abalone_regression Model
    if recreate or not Model("abalone-regression-100").exists():
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
            name="abalone-regression-100",
            model_type=ModelType.REGRESSOR,
            feature_list=features,
            target_column="class_number_of_rings",
            tags=["abalone", "regression"],
            description="Abalone Regression Model",
            train_all_data=True,
        )
        m.set_owner("test")

    # Create the Endpoint
    if recreate or not Endpoint("abalone-regression-100").exists():
        model = Model("abalone-regression-100")
        end = model.to_endpoint(tags=["abalone", "regression", "100", "test"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create the aqsol solubility regression Model
    if recreate or not Model("aqsol-regression-100").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        feature_list = [
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
        feature_set.to_model(
            name="aqsol-regression-100",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=feature_list,
            description="AQSol Regression Model",
            tags=["aqsol", "regression"],
        )

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-regression-100").exists():
        m = Model("aqsol-regression-100")
        m.to_endpoint(tags=["aqsol", "regression", "100", "test"])