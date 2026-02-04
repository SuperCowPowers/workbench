"""Create the SMILES to 3D Molecular Descriptors Model + Endpoint
   Note: This version REMOVES salts in the molecules.

Description:
   Feature Endpoint that takes a SMILES string and computes 3D conformer-based
   molecular descriptors including:
   - RDKit 3D shape descriptors (PMI, NPR, asphericity, etc.)
   - Mordred 3D descriptors (CPSA, GeometricalIndex, GravitationalIndex, PBF)
   - Pharmacophore 3D descriptors (amphiphilic moment, IMHB potential, etc.)
   - Conformer ensemble statistics (energy, flexibility)

   Total: 75 3D descriptors

   Note: This endpoint is slower than the 2D descriptor endpoint due to
   conformer generation. Recommended for batch processing rather than
   real-time inference.

Created Artifacts:

    Models:
        - smiles-to-3d-descriptors-v1

    Endpoints:
        - smiles-to-3d-descriptors-v1
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
    script_path = "model_scripts/rdkit_3d_model_script.py"

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
    tags = ["smiles", "3d descriptors", "conformer", "pharmacophore", "shape"]
    model = feature_set.to_model(
        name="smiles-to-3d-descriptors-v1",
        model_type=ModelType.TRANSFORMER,
        feature_list=["smiles"],
        description="SMILES to 3D Molecular Descriptors (75 features: shape, CPSA, pharmacophore, conformer stats)",
        tags=tags,
        custom_script=script_path,
    )
    model.set_owner("BW")

    # Create the endpoint for the model
    # Note: 3D descriptor computation is more memory/CPU intensive
    # Using higher memory for serverless, larger instance for non-serverless
    if serverless:
        end = model.to_endpoint(tags=tags, serverless=True, mem_size=4096, max_concurrency=5)
    else:
        end = model.to_endpoint(tags=tags, serverless=False, instance="ml.c7i.large")

    # Set smaller batch size for 3D endpoint (conformer generation is slow ~1-1.5s per molecule)
    # At ~0.7 mol/s, 20 molecules takes ~28s, safely under the 60s serverless timeout
    end.upsert_workbench_meta({"inference_batch_size": 20})

    # Run inference on the endpoint (smaller batch due to slower processing)
    end.inference(feature_set.pull_dataframe()[:50])
