"""This Script creates the Abalone and AQSol Quantile Regression Artifacts in AWS/Workbench

FeatureSets:
    - test-quantile
Models:
    - test-quantile
    - abalone-quantile
    - aqsol-quantile
Endpoints:
    - test-quantile
    - abalone-quantile
    - aqsol-quantile
"""

import logging

from workbench.api import FeatureSet, Model, ModelType, Endpoint
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.plot_utils import generate_heteroskedastic_data

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the Test Quantile Regression FeatureSet
    if recreate or not FeatureSet("test_quantile").exists():
        # Generate a synthetic dataset with heteroskedasticity
        df = generate_heteroskedastic_data(n=1000)
        to_features = PandasToFeatures("test_quantile")
        to_features.set_output_tags(["test", "quantiles"])
        to_features.set_input(df, id_column="id")
        to_features.transform()

    # Create the Test Quantile Regression Model
    if recreate or not Model("test-quantile").exists():
        # Transform FeatureSet into Quantile Regression Model
        feature_set = FeatureSet("test_quantile")
        feature_set.to_model(
            name="test-quantile",
            model_type=ModelType.UQ_REGRESSOR,
            target_column="y",
            description="Test Quantile Regression",
            tags=["test", "quantiles"],
        )

    # Create the Test Quantile Regression Endpoint
    if recreate or not Endpoint("test-quantile").exists():
        m = Model("test-quantile")
        end = m.to_endpoint(tags=["test", "quantiles"])

        # Run auto-inference on the Test Quantile Regression Endpoint
        end.auto_inference(capture=True)

    # Create the Abalone Quantile Regression Model
    if recreate or not Model("abalone-quantile").exists():

        # Transform FeatureSet into Quantile Regression Model
        feature_set = FeatureSet("abalone_features")
        feature_set.to_model(
            name="abalone-quantile",
            model_type=ModelType.UQ_REGRESSOR,
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
    if recreate or not Model("aqsol-quantiles").exists():

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
            name="aqsol-quantiles",
            model_type=ModelType.UQ_REGRESSOR,
            target_column="solubility",
            feature_list=features,
            description="AQSol Quantile Regression",
            tags=["aqsol", "quantiles"],
        )

    # Create the AQSol Quantile Regression Endpoint
    if recreate or not Endpoint("aqsol-quantiles").exists():
        m = Model("aqsol-quantiles")
        end = m.to_endpoint(tags=["aqsol", "quantiles"])

        # Run auto-inference on the AQSol Quantile Regression Endpoint
        end.auto_inference(capture=True)
