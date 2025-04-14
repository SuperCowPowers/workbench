"""This Script creates MetaEndpoints in AWS/Workbench

Models:
    - abalone-regression-meta

Endpoints:
    - abalone-regression-meta
"""

import logging
import pandas as pd
import awswrangler as wr

from workbench.api import DataSource, FeatureSet, Model, ModelType, Endpoint
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.model_utils import get_custom_script_path

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Get the path to the dataset in S3
    s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = True

    # Create the abalone_regression Model
    tags = ["abalone", "regression", "meta"]
    if recreate or not Model("abalone-regression-meta").exists():
        script_path = get_custom_script_path("meta_endpoints", "example.py")
        fs = FeatureSet("abalone_features")
        m = fs.to_model(
            name="abalone-regression-meta",
            model_type=ModelType.REGRESSOR,
            target_column="class_number_of_rings",
            tags=tags,
            description="Abalone Regression Meta Model",
            custom_script=script_path,
            inference_image="workbench-inference",
        )
        m.set_owner("test")

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression-meta").exists():
        model = Model("abalone-regression-meta")
        end = model.to_endpoint(tags=tags)

        # Run inference on the endpoint
        end.auto_inference(capture=True)
