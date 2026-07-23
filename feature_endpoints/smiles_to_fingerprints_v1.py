"""Create the SMILES → Morgan Count Fingerprint Feature Endpoint.

Takes a SMILES string and computes Morgan count fingerprints (4096-dim,
radius 2 / ECFP4 equivalent). Each position holds the count of that circular
substructure, clamped to uint8 — suited to ADMET modeling and to similarity /
proximity workflows. Salts are handled internally via largest-fragment selection.

Output is a single ``fingerprint`` column (comma-separated uint8 counts).

Created artifacts:  Model/Endpoint ``smiles-to-fingerprints-v1``
"""

import os
from workbench.api import ModelType, ModelFramework
from _common import ensure_featureset

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
# These are the settings you'll most often want to tweak. Everything else
# comes from workbench defaults.
ENDPOINT_NAME = "smiles-to-fingerprints-v1"
# SERVERLESS=True → AWS-managed scaling via max_concurrency; cheap idle.
# SERVERLESS=False → dedicated instance (more predictable latency).
SERVERLESS = os.environ.get("SERVERLESS", "True").lower() == "true"
MEM_SIZE = 4096  # MB — serverless memory ceiling.
MAX_CONCURRENCY = 5  # serverless concurrent invocations.
INSTANCE = "ml.c7i.large"  # used only when SERVERLESS=False.


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    feature_set = ensure_featureset()
    tags = ["smiles", "fingerprints", "morgan", "ecfp"]
    model = feature_set.to_model(
        name=ENDPOINT_NAME,
        model_type=ModelType.TRANSFORMER,
        model_framework=ModelFramework.TRANSFORMER,
        feature_list=["smiles"],
        description="SMILES to Morgan count fingerprints (4096-dim, radius 2 / ECFP4)",
        tags=tags,
        custom_script="model_scripts/smiles_to_fingerprints_model_script.template",
        hyperparameters={"radius": 2, "n_bits": 4096, "counts": True},
    )
    model.set_owner("BW")

    # ── Deploy as a realtime endpoint (serverless or dedicated instance).
    # Fingerprints compute fast, so serverless is almost always the right choice —
    # scale-to-zero idle cost, AWS-managed concurrency.
    if SERVERLESS:
        model.to_endpoint(tags=tags, serverless=True, mem_size=MEM_SIZE, max_concurrency=MAX_CONCURRENCY)
    else:
        model.to_endpoint(tags=tags, serverless=False, instance=INSTANCE)
