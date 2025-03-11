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
    if recreate or not Model("abalone-quantile").exists():

        # Transform FeatureSet into Quantile Regression Model
        feature_set = FeatureSet("abalone_features")
        feature_set.to_model(
            name="abalone-quantile",
            model_type=ModelType.QUANTILE_REGRESSOR,
            target_column="class_number_of_rings",
            description="Abalone Quantile Regression",
            tags=["abalone", "quantiles"],
        )

    # Create the Abalone Quantile Regression Endpoint
    if recreate or not Endpoint("abalone-quantile").exists():
        m = Model("abalone-quantile")
        end = m.to_endpoint(tags=["abalone", "quantiles"])

        # Run auto-inference on the Abalone Quantile Regression Endpoint
        end.auto_inference(capture=True)

    # Create the AQSol Quantile Regression Model
    if recreate or not Model("aqsol-quantile").exists():

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
            name="aqsol-quantile",
            model_type=ModelType.QUANTILE_REGRESSOR,
            target_column="solubility",
            feature_list=features,
            description="AQSol Quantile Regression",
            tags=["aqsol", "quantiles"],
        )

    # Create the AQSol Quantile Regression Endpoint
    if recreate or not Endpoint("aqsol-quantile").exists():
        m = Model("aqsol-quantile")
        end = m.to_endpoint(tags=["aqsol", "quantiles"])

        # Run auto-inference on the AQSol Quantile Regression Endpoint
        end.auto_inference(capture=True)
