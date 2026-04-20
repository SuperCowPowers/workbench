"""Create the SMILES → 3D Molecular Descriptors (Boltzmann) Async Endpoint.

Salts are removed. Computes 74 Boltzmann-weighted 3D descriptors per SMILES:
  - RDKit 3D shape      (PMI, NPR, asphericity, …)
  - Mordred 3D          (CPSA, GeometricalIndex, GravitationalIndex, PBF)
  - Pharmacophore 3D    (amphiphilic moment, IMHB potential, …)
  - Conformer ensemble  (energy, flexibility)

Uses adaptive conformer counts (50–300 based on rotatable bonds) and
Boltzmann-weighted ensemble averaging. Deployed as an async endpoint
(up to 15-minute invocations) for overnight batch processing of 10k–100k
compound libraries.

Created artifacts:  Model/Endpoint ``smiles-to-3d-full-v1``
"""

from workbench.api import ModelType, ModelFramework
from workbench.utils.feature_endpoint_utils import ensure_demo_featureset, register_features

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
# These are the settings you'll most often want to tweak. Everything else
# comes from workbench defaults.
ENDPOINT_NAME = "smiles-to-3d-full-v1"
INSTANCE = None  # None → auto-select (ml.c7i.xlarge for async). Set
#        "ml.c7i.2xlarge" etc. for more CPU/mem per worker.
MAX_INSTANCES = 8  # Autoscaler ceiling. Bump for bigger batch jobs.
IDLE_MINUTES = 15  # Minutes of empty queue before draining to zero.


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    feature_set = ensure_demo_featureset()
    tags = ["smiles", "3d descriptors", "boltzmann", "conformer ensemble"]
    model = feature_set.to_model(
        name=ENDPOINT_NAME,
        model_type=ModelType.TRANSFORMER,
        model_framework=ModelFramework.TRANSFORMER,
        feature_list=["smiles"],
        description="SMILES to 3D Molecular Descriptors — Boltzmann ensemble (74 features)",
        tags=tags,
        custom_script="model_scripts/smiles_to_3d_full_model_script.py",
    )
    model.set_owner("BW")

    # Advanced: workbench_meta knobs set BEFORE to_endpoint() shape deployment.
    #   model.upsert_workbench_meta({"async_max_concurrent_per_instance": 4})
    #     - default 1 (good for CPU-saturating work). Bump for IO-bound inference.

    # ── Deploy as an async (batch) endpoint.
    # Autoscaler runs in "batch" mode: 0 → MAX_INSTANCES in one step on first
    # traffic, drain to 0 after IDLE_MINUTES of empty queue.
    end = model.to_endpoint(
        tags=tags,
        async_endpoint=True,
        instance=INSTANCE,
        max_instances=MAX_INSTANCES,
        scale_in_idle_minutes=IDLE_MINUTES,
    )

    # Advanced: workbench_meta knobs set AFTER to_endpoint() tune runtime inference.
    #   end.upsert_workbench_meta({"inference_batch_size": 100})
    #     - default 50. Higher = better overhead amortization, but a single chunk
    #       must finish inside SageMaker's 1hr async invocation limit.
    #   end.upsert_workbench_meta({"inference_max_in_flight": 32})
    #     - default 16. Higher = more backlog pressure on autoscaling, more S3 load.

    # Register output columns to ParameterStore at
    # /workbench/feature_lists/<endpoint_name>. Also smoke-tests the endpoint.
    register_features(end)
