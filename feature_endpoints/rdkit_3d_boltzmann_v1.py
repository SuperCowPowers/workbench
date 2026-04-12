"""Create the SMILES to 3D Molecular Descriptors (Boltzmann) Model + Endpoint
   Note: This version REMOVES salts in the molecules.

Description:
   Async Feature Endpoint that takes a SMILES string and computes Boltzmann-
   weighted 3D conformer-based molecular descriptors including:
   - RDKit 3D shape descriptors (PMI, NPR, asphericity, etc.)
   - Mordred 3D descriptors (CPSA, GeometricalIndex, GravitationalIndex, PBF)
   - Pharmacophore 3D descriptors (amphiphilic moment, IMHB potential, etc.)
   - Conformer ensemble statistics (energy, flexibility)

   Total: 75 3D descriptors (same features as the fast v1 endpoint)

   This endpoint uses adaptive conformer counts (50-300 based on rotatable
   bonds) and Boltzmann-weighted ensemble averaging instead of the single
   lowest-energy conformer used by the fast endpoint. Designed for overnight
   batch processing of 10k-100k compound libraries.

   Deployed as an async SageMaker endpoint (up to 15-minute invocations).

Created Artifacts:

    Models:
        - smiles-to-3d-boltzmann-v1

    Endpoints:
        - smiles-to-3d-boltzmann-v1
"""

import os

# Workbench Imports
from workbench.api import FeatureSet, Model, ModelType, ModelFramework, PublicData
from workbench.core.transforms.pandas_transforms import PandasToFeatures

# Async endpoint on a realtime instance — no serverless option for async.
# 8 vCPUs lets RDKit's numThreads=0 parallelize conformer embedding.
DEFAULT_INSTANCE = "ml.c7i.2xlarge"

if __name__ == "__main__":

    instance = os.environ.get("INSTANCE", DEFAULT_INSTANCE)

    # Pull in the custom script path
    script_path = "model_scripts/rdkit_3d_boltzmann_model_script.py"

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
    tags = ["smiles", "3d descriptors", "boltzmann", "conformer ensemble"]
    RECREATE = True
    if RECREATE:
        model = feature_set.to_model(
            name="smiles-to-3d-boltzmann-v1",
            model_type=ModelType.TRANSFORMER,
            model_framework=ModelFramework.TRANSFORMER,
            feature_list=["smiles"],
            description="SMILES to 3D Molecular Descriptors — Boltzmann ensemble (75 features)",
            tags=tags,
            custom_script=script_path,
        )
        model.set_owner("BW")

    # Deploy as async endpoint — batch_size=1 because each molecule can take minutes
    model = Model("smiles-to-3d-boltzmann-v1")
    end = model.to_endpoint(tags=tags, async_endpoint=True, instance=instance)
    end.upsert_workbench_meta({"inference_batch_size": 1})
    print(f"Async endpoint deployed: {end.name} on {instance}")

    # Quick smoke test with a small batch
    end.inference(feature_set.pull_dataframe()[:5])
