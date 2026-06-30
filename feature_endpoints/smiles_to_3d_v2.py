"""Create the SMILES → 3D Molecular Descriptors v2 (curated, xTB) Async Endpoint.

Salts are removed. Computes 26 Boltzmann-weighted v2 3D descriptors per SMILES —
a deliberately small set chosen to add signal orthogonal to the 2D descriptors
(the v1 "full" 74-feature set never beat 2D, even on non-PXR ADMET assays):

  - Electronic (xTB)    dipole, quadrupole, HOMO/LUMO/gap, hardness,
                        electrophilicity, partial-charge summaries — harvested
                        from the SAME GFN2-xTB single point used for ranking
  - Surface             SASA total/polar/apolar/fraction + charge-weighted PSA
  - Shape               NPR1/NPR2, asphericity, Rg, spherocity, PBF
  - Pharmacophore geom  amphiphilic moment, charge/HBA centroid offsets
  - Flexibility         conformational-flexibility index, confs in window

Same conformer engine as smiles-to-3d-full-v1 (ETKDGv3 → MMFF opt → GFN2-xTB
ranking → Boltzmann weighting); only the descriptor layer differs. Deployed as
an async endpoint (up to 60-minute invocations) for batch processing.

Created artifacts:  Model/Endpoint ``smiles-to-3d-v2``
"""

from workbench.api import ModelType, ModelFramework
from _common import ensure_featureset

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
ENDPOINT_NAME = "smiles-to-3d-v2"
INSTANCE = None  # None → auto-select (ml.c7i.xlarge for async).
MIN_INSTANCES = 0  # Autoscaler floor. 0 in dev (scale to zero); 1 in prod.
MAX_INSTANCES = 16  # Autoscaler ceiling. Bump for bigger batch jobs.
BATCH_SIZE = 4  # Rows per invocation. Smaller = steadier under load (shorter jobs,
#                 finer failure granularity); a MetaEndpoint wrapping this sizes its
#                 own batch off BATCH_SIZE × MAX_INSTANCES.


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    feature_set = ensure_featureset()
    tags = ["smiles", "3d descriptors", "v2", "curated", "xtb", "conformer ensemble"]
    model = feature_set.to_model(
        name=ENDPOINT_NAME,
        model_type=ModelType.TRANSFORMER,
        model_framework=ModelFramework.TRANSFORMER,
        feature_list=["smiles"],
        description="SMILES to 3D Molecular Descriptors v2 — curated xTB-powered set (26 features)",
        tags=tags,
        custom_script="model_scripts/smiles_to_3d_v2_model_script.py",
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
