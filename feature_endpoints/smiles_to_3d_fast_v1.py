"""Create the SMILES to 3D Molecular Descriptors Model + Endpoint
   Note: This version REMOVES salts in the molecules.

Description:
   Feature Endpoint that takes a SMILES string and computes 3D conformer-based
   molecular descriptors including:
   - RDKit 3D shape descriptors (PMI, NPR, asphericity, etc.)
   - Mordred 3D descriptors (CPSA, GeometricalIndex, GravitationalIndex, PBF)
   - Pharmacophore 3D descriptors (amphiphilic moment, IMHB potential, etc.)
   - Conformer ensemble statistics (energy, flexibility)

   Total: 74 3D descriptors

   Note: This endpoint is slower than the 2D descriptor endpoint due to
   conformer generation. Recommended for batch processing rather than
   real-time inference.

Created Artifacts:

    Models:
        - smiles-to-3d-fast-v1

    Endpoints:
        - smiles-to-3d-fast-v1
"""

import os

# Workbench Imports
from workbench.api import FeatureSet, Model, ModelType, ModelFramework, PublicData
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.feature_endpoint_utils import register_features

# Inference batch size tuned per deployment config. 3D conformer generation is
# CPU-heavy, so the ideal batch size scales with available vCPUs.
BATCH_SIZE_BY_CONFIG = {
    "serverless": 5,  # ~2 vCPUs, 2-6GB memory
    "ml.c7i.xlarge": 10,  # 4 vCPUs, 8 GB
    "ml.c7i.2xlarge": 20,  # 8 vCPUs, 16 GB
}

# Default realtime instance when SERVERLESS=false
DEFAULT_INSTANCE = "ml.c7i.xlarge"

if __name__ == "__main__":

    serverless = os.environ.get("SERVERLESS", "True").lower() == "true"
    instance = os.environ.get("INSTANCE", DEFAULT_INSTANCE)

    # Pull in the custom script path
    script_path = "model_scripts/smiles_to_3d_fast_model_script.py"

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
            name="smiles-to-3d-fast-v1",
            model_type=ModelType.TRANSFORMER,
            model_framework=ModelFramework.TRANSFORMER,
            feature_list=["smiles"],
            description="SMILES to 3D Molecular Descriptors (74 features: shape, CPSA, pharmacophore, conformer stats)",
            tags=tags,
            custom_script=script_path,
        )
        model.set_owner("BW")

    # Deploy the endpoint and pick the batch size for this config
    model = Model("smiles-to-3d-fast-v1")
    if serverless:
        end = model.to_endpoint(tags=tags, serverless=True, mem_size=6144, max_concurrency=5)
        batch_size = BATCH_SIZE_BY_CONFIG["serverless"]
    else:
        end = model.to_endpoint(tags=tags, serverless=False, instance=instance)
        batch_size = BATCH_SIZE_BY_CONFIG.get(instance, BATCH_SIZE_BY_CONFIG[DEFAULT_INSTANCE])

    end.upsert_workbench_meta({"inference_batch_size": batch_size})
    print(f"inference_batch_size={batch_size}")

    # Register output feature columns to ParameterStore at
    # /workbench/feature_lists/<endpoint_name> (also smoke-tests the endpoint).
    register_features(end)
