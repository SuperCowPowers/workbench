"""This Script creates the AQSol Ensemble Regression Artifacts in AWS/Workbench

Models:
    - aqsol-ensemble
Endpoints:
    - aqsol-ensemble
"""

import logging

from workbench.api import FeatureSet, Model, ModelType, Endpoint

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = True

    # Create the AQSol Ensemble Regression Model
    if recreate or not Model("aqsol-ensemble").exists():

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
            name="aqsol-ensemble",
            model_type=ModelType.ENSEMBLE_REGRESSOR,
            target_column="solubility",
            feature_list=features,
            description="AQSol Ensemble Regression",
            tags=["aqsol", "ensemble"],
        )

    # Create the AQSol Quantile Regression Endpoint
    if recreate or not Endpoint("aqsol-ensemble").exists():
        m = Model("aqsol-ensemble")
        end = m.to_endpoint(tags=["aqsol", "ensemble"])

        # Run auto-inference on the AQSol Ensemble Regression Endpoint
        end.auto_inference(capture=True)
