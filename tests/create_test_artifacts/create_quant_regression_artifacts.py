"""This Script creates the Abalone and AQSol Quantile Regression Artifacts in AWS/Workbench

Models:
    - abalone-quantile-reg
    - aqsol-quantile-reg
Endpoints:
    - abalone-qr-end
    - aqsol-qr-end
"""

import logging
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.api.endpoint import Endpoint

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the Abalone Quantile Regression Model
    if recreate or not Model("abalone-quantile-reg").exists():

        # Transform FeatureSet into Quantile Regression Model
        feature_set = FeatureSet("abalone_features")
        feature_set.to_model(
            name="abalone-quantile-reg",
            model_type=ModelType.REGRESSOR,
            target_column="class_number_of_rings",
            description="Abalone Quantile Regression",
            tags=["abalone", "quantiles"],
        )

    # Create the Abalone Quantile Regression Endpoint
    if recreate or not Endpoint("abalone-qr").exists():
        m = Model("abalone-quantile-reg")
        end = m.to_endpoint(name="abalone-qr", tags=["abalone", "quantiles"])

        # Run auto-inference on the Abalone Quantile Regression Endpoint
        end.auto_inference(capture=True)

    # Create the AQSol Quantile Regression Model
    if recreate or not Model("aqsol-quantile-reg").exists():

        # AQSol Features
        features = [
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

        # Transform FeatureSet into Quantile Regression Model
        feature_set = FeatureSet("aqsol_features")
        feature_set.to_model(
            name="aqsol-quantile-reg",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=features,
            description="AQSol Quantile Regression",
            tags=["aqsol", "quantiles"],
        )

    # Create the AQSol Quantile Regression Endpoint
    if recreate or not Endpoint("aqsol-qr").exists():
        m = Model("aqsol-quantile-reg")
        end = m.to_endpoint(name="aqsol-qr", tags=["aqsol", "quantiles"])

        # Run auto-inference on the AQSol Quantile Regression Endpoint
        end.auto_inference(capture=True)
