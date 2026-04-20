"""Create the SMILES → Standardize/Tautomer + RDKit + Mordred Feature Endpoint.

Salts are removed. Takes a SMILES string and computes RDKit + Mordred 2D
molecular descriptors (313 features total).

Created artifacts:  Model/Endpoint ``smiles-to-2d-v1``
"""

import os

from workbench.api import ModelType, ModelFramework
from workbench.utils.feature_endpoint_utils import ensure_demo_featureset, register_features

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
# These are the settings you'll most often want to tweak. Everything else
# comes from workbench defaults.
ENDPOINT_NAME = "smiles-to-2d-v1"
# SERVERLESS=True → AWS-managed scaling via max_concurrency; cheap idle.
# SERVERLESS=False → dedicated instance (more predictable latency).
SERVERLESS = os.environ.get("SERVERLESS", "True").lower() == "true"
MEM_SIZE = 4096  # MB — serverless memory ceiling.
MAX_CONCURRENCY = 5  # serverless concurrent invocations.
INSTANCE = "ml.c7i.large"  # used only when SERVERLESS=False.


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    feature_set = ensure_demo_featureset()
    tags = ["smiles", "molecular descriptors", "tautomerized", "stereo"]
    model = feature_set.to_model(
        name=ENDPOINT_NAME,
        model_type=ModelType.TRANSFORMER,
        model_framework=ModelFramework.TRANSFORMER,
        feature_list=["smiles"],
        description="SMILES to RDKit + Mordred 2D Molecular Descriptors (salts removed)",
        tags=tags,
        custom_script="model_scripts/smiles_to_2d_model_script.py",
    )
    model.set_owner("BW")

    # ── Deploy as a realtime endpoint (serverless or dedicated instance).
    # 2D descriptors compute fast enough that serverless is almost always the
    # right choice — scale-to-zero idle cost, AWS-managed concurrency.
    if SERVERLESS:
        end = model.to_endpoint(tags=tags, serverless=True, mem_size=MEM_SIZE, max_concurrency=MAX_CONCURRENCY)
    else:
        end = model.to_endpoint(tags=tags, serverless=False, instance=INSTANCE)

    # Register output columns to ParameterStore at
    # /workbench/feature_lists/<endpoint_name>. Also smoke-tests the endpoint.
    register_features(end)
