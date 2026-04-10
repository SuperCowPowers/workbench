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

# Workbench Imports
from workbench.api import FeatureSet, Model, ModelType, PublicData
from workbench.core.transforms.pandas_transforms import PandasToFeatures

if __name__ == "__main__":

    # Serverless and Instance types
    serverless = os.environ.get("SERVERLESS", "True").lower() == "true"

    # Pull in the custom script path
    script_path = "model_scripts/rdkit_3d_model_script.py"

    # Check if we have an existing FeatureSet, if not create one
    if not FeatureSet("feature_endpoint_fs").exists():

        # Grab the public AqSol data (contains SMILES and experimental solubility values)
        aqsol_data = PublicData().get("comp_chem/aqsol/aqsol_public_data")
        aqsol_data.columns = aqsol_data.columns.str.lower()

        to_features = PandasToFeatures("feature_endpoint_fs")
        to_features.set_input(aqsol_data, id_column="id")
        to_features.set_output_tags(["aqsol", "public"])
        to_features.transform()
        fs = FeatureSet("feature_endpoint_fs")
        fs.set_owner("FeatureEndpoint")

    # Grab our FeatureSet and create the Model
    feature_set = FeatureSet("feature_endpoint_fs")
    tags = ["smiles", "3d descriptors", "conformer", "pharmacophore", "shape"]
    RECREATE = True
    if RECREATE:
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
    # Note: 3D descriptor computation is CPU-intensive (conformer generation + force field optimization)
    # Realtime instance recommended — serverless has a hard 60s timeout that's too tight for
    # complex molecules that can take 30-40s each on serverless hardware.
    model = Model("smiles-to-3d-descriptors-v1")
    if serverless:
        end = model.to_endpoint(tags=tags, serverless=True, mem_size=6144, max_concurrency=5)
        end.upsert_workbench_meta({"inference_batch_size": 5})
    else:
        end = model.to_endpoint(tags=tags, serverless=False, instance="ml.c7i.2xlarge")
        end.upsert_workbench_meta({"inference_batch_size": 10})

    # Run inference on the endpoint (smaller batch due to slower processing)
    end.inference(feature_set.pull_dataframe()[:50])
