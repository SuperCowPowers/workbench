"""Create the SMILES → 3D Molecular Descriptors (Boltzmann) Async Endpoint.

Salts are removed. Computes 74 Boltzmann-weighted 3D descriptors per SMILES:
  - RDKit 3D shape      (PMI, NPR, asphericity, …)
  - Mordred 3D          (CPSA, GeometricalIndex, GravitationalIndex, PBF)
  - Pharmacophore 3D    (amphiphilic moment, IMHB potential, …)
  - Conformer ensemble  (energy, flexibility)

Uses adaptive conformer counts (50–300 based on rotatable bonds) and
Boltzmann-weighted ensemble averaging. Deployed as an async endpoint
(up to 60-minute invocations) for overnight batch processing of 10k–100k
compound libraries.

Created artifacts:  Model/Endpoint ``smiles-to-3d-v1``
"""

from workbench.api import ModelType, ModelFramework
from _common import ensure_featureset

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
# These are the settings you'll most often want to tweak. Everything else
# comes from workbench defaults.
ENDPOINT_NAME = "smiles-to-3d-v1"
INSTANCE = None  # None → auto-select (ml.c7i.xlarge for async). Set
#        "ml.c7i.2xlarge" etc. for more CPU/mem per worker.
MIN_INSTANCES = 0  # Autoscaler floor. 0 in dev (scale to zero); set to 1 in
#         production to keep one instance warm.
MAX_INSTANCES = 8  # Autoscaler ceiling. Bump for bigger batch jobs.
BATCH_SIZE = 4  # Rows per invocation. Smaller = steadier under load (shorter jobs,
#         finer failure granularity). Fits ml.c7i.xlarge; bump on ml.c7i.2xlarge.


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    feature_set = ensure_featureset()
    tags = ["smiles", "3d descriptors", "full", "conformer ensemble"]
    model = feature_set.to_model(
        name=ENDPOINT_NAME,
        model_type=ModelType.TRANSFORMER,
        model_framework=ModelFramework.TRANSFORMER,
        feature_list=["smiles"],
        description="SMILES to 3D Molecular Descriptors — Boltzmann ensemble (74 features)",
        tags=tags,
        custom_script="model_scripts/smiles_to_3d_v1_model_script.py",
    )
    model.set_owner("BW")

    # ── Deploy as an async (batch) endpoint.
    end = model.to_endpoint(
        tags=tags,
        async_endpoint=True,
        instance=INSTANCE,
        min_instances=MIN_INSTANCES,
        max_instances=MAX_INSTANCES,
    )

    # Per-invocation batch size (overrides the workbench default of 10). Also stamp
    # the fleet ceiling so a MetaEndpoint wrapping this child can size its batch to
    # the fleet (smallest async batch × max_instances × 4).
    end.upsert_workbench_meta({"inference_batch_size": BATCH_SIZE, "max_instances": MAX_INSTANCES})
