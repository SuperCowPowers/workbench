"""Create the SMILES to Standardize/Tautomer + RDKIT + Mordred Model + Endpoint
   Note: This version KEEPS salts in the molecules.

Description:
   Feature Endpoint that takes a SMILES string and
   computes RDKIT + Mordred Molecular Descriptors

Created Artifacts:

    Models:
        - smiles-to-taut-md-stereo-v1-keep-salts

    Endpoints:
        - smiles-to-taut-md-stereo-v1-keep-salts
"""

import os
import awswrangler as wr

# Workbench Imports
from workbench.api import FeatureSet, ModelType
from workbench.core.transforms.pandas_transforms import PandasToFeatures

if __name__ == "__main__":

    # Serverless and Instance types
    serverless = os.environ.get("SERVERLESS", "True").lower() == "true"

    # Pull in the custom script path
    script_path = "model_scripts/rdkit_mordred_model_script_keep_salts.py"

    # Check if we have an existing FeatureSet, if not create one
    if not FeatureSet("feature_endpoint_fs").exists():
        s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"
        aqsol_data = wr.s3.read_csv(s3_path)
        aqsol_data.columns = aqsol_data.columns.str.lower()

        to_features = PandasToFeatures("feature_endpoint_fs")
        to_features.set_input(aqsol_data, id_column="id")
        to_features.set_output_tags(["aqsol", "public"])
        to_features.transform()
        fs = FeatureSet("feature_endpoint_fs")
        fs.set_owner("FeatureEndpoint")

    # Grab our FeatureSet
    feature_set = FeatureSet("feature_endpoint_fs")
    tags = ["smiles", "molecular descriptors", "tautomerized", "stereo"]
    model = feature_set.to_model(
        name="smiles-to-taut-md-stereo-v1-keep-salts",
        model_type=ModelType.TRANSFORMER,
        feature_list=["smiles"],
        description="Smiles to Molecular Descriptors",
        tags=tags,
        custom_script=script_path,
    )
    model.set_owner("BW")

    # Create the endpoint for the model
    if serverless:
        end = model.to_endpoint(tags=tags, serverless=True, mem_size=4096, max_concurrency=5)
    else:
        end = model.to_endpoint(tags=tags, serverless=False, instance="ml.c7i.large")

    # Run inference on the endpoint
    end.inference(feature_set.pull_dataframe()[:100])
