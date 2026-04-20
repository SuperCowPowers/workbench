"""Create the SMILES → 3D Molecular Descriptors (fast) Feature Endpoint.

Salts are removed. Computes 74 3D descriptors per SMILES:
  - RDKit 3D shape      (PMI, NPR, asphericity, …)
  - Mordred 3D          (CPSA, GeometricalIndex, GravitationalIndex, PBF)
  - Pharmacophore 3D    (amphiphilic moment, IMHB potential, …)
  - Conformer ensemble  (energy, flexibility)

Uses the single lowest-energy conformer (fast). For Boltzmann-weighted
ensemble descriptors on the same feature set, use the async endpoint
``smiles-to-3d-full-v1`` instead — slower per-row but more accurate for
flexible molecules.

Created artifacts:  Model/Endpoint ``smiles-to-3d-fast-v1``
"""

import os

from workbench.api import Model, ModelType, ModelFramework
from workbench.utils.feature_endpoint_utils import ensure_demo_featureset, register_features

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
# These are the settings you'll most often want to tweak. Everything else
# comes from workbench defaults.
ENDPOINT_NAME = "smiles-to-3d-fast-v1"
SERVERLESS = os.environ.get("SERVERLESS", "True").lower() == "true"
INSTANCE = os.environ.get("INSTANCE", "ml.c7i.xlarge")  # only used when SERVERLESS=False
MEM_SIZE = 6144  # MB — serverless memory ceiling (3D needs more than 2D).
MAX_CONCURRENCY = 5  # serverless concurrent invocations.
RECREATE_MODEL = True  # set False to skip model re-create if only redeploying the endpoint

# Inference batch size tuned per deployment config. 3D conformer generation is
# CPU-heavy, so the ideal batch size scales with available vCPUs.
BATCH_SIZE_BY_CONFIG = {
    "serverless": 5,  # ~2 vCPUs, 2-6GB memory
    "ml.c7i.xlarge": 10,  # 4 vCPUs, 8 GB
    "ml.c7i.2xlarge": 20,  # 8 vCPUs, 16 GB
}


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    feature_set = ensure_demo_featureset()
    tags = ["smiles", "3d descriptors", "conformer", "pharmacophore", "shape"]
    if RECREATE_MODEL:
        model = feature_set.to_model(
            name=ENDPOINT_NAME,
            model_type=ModelType.TRANSFORMER,
            model_framework=ModelFramework.TRANSFORMER,
            feature_list=["smiles"],
            description="SMILES to 3D Molecular Descriptors (74 features: shape, CPSA, pharmacophore, conformer stats)",
            tags=tags,
            custom_script="model_scripts/smiles_to_3d_fast_model_script.py",
        )
        model.set_owner("BW")

    # ── Deploy as a realtime endpoint and tune runtime batch size for this config.
    model = Model(ENDPOINT_NAME)
    if SERVERLESS:
        end = model.to_endpoint(tags=tags, serverless=True, mem_size=MEM_SIZE, max_concurrency=MAX_CONCURRENCY)
        batch_size = BATCH_SIZE_BY_CONFIG["serverless"]
    else:
        end = model.to_endpoint(tags=tags, serverless=False, instance=INSTANCE)
        batch_size = BATCH_SIZE_BY_CONFIG.get(INSTANCE, BATCH_SIZE_BY_CONFIG["ml.c7i.xlarge"])

    end.upsert_workbench_meta({"inference_batch_size": batch_size})
    print(f"inference_batch_size={batch_size}")

    # Register output columns to ParameterStore at
    # /workbench/feature_lists/<endpoint_name>. Also smoke-tests the endpoint.
    register_features(end)
