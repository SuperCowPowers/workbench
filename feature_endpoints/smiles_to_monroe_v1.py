"""Create the SMILES → Monroe Molecular Embedding Feature Endpoint.

Salts are removed. Takes a SMILES string and computes the 720-d embedding of the
frozen Monroe encoder (columns ``monroe_000``..``monroe_719``). The pretrained weights
come from the foundation checkpoint registry and travel in the model artifact, so stage
them first with ``scripts/admin/push_foundation_models.py --model monroe``.

Created artifacts:  Model/Endpoint ``smiles-to-monroe-v1``
"""

from workbench.api import ModelType, ModelFramework, PublicData
from _common import ensure_featureset

# ─── Deploy-time knobs ──────────────────────────────────────────────────────
ENDPOINT_NAME = "smiles-to-monroe-v1"
MEM_SIZE = 6144  # MB — serverless memory ceiling (and the most vCPU serverless gives).
MAX_CONCURRENCY = 8  # serverless concurrent invocations.
BATCH_SIZE = 5  # Rows per invocation: a slow molecule can take 10 s, a request gets 60 s.


if __name__ == "__main__":
    # ── Create the Model (shared AqSol-backed demo FeatureSet as training source).
    # PYTORCH picks the pytorch_chem images (the encoder needs torch).
    feature_set = ensure_featureset()
    tags = ["smiles", "monroe", "embedding", "foundation model"]
    model = feature_set.to_model(
        name=ENDPOINT_NAME,
        model_type=ModelType.TRANSFORMER,
        model_framework=ModelFramework.PYTORCH,
        feature_list=["smiles"],
        description="SMILES to Monroe Molecular Embedding (720 features, salts removed)",
        tags=tags,
        custom_script="model_scripts/smiles_to_monroe_model_script.py",
    )
    model.set_owner("BW")

    # ── Deploy as a serverless endpoint.
    end = model.to_endpoint(tags=tags, serverless=True, mem_size=MEM_SIZE, max_concurrency=MAX_CONCURRENCY)
    end.upsert_workbench_meta({"inference_batch_size": BATCH_SIZE})

    # Smoke test with a few public compounds.
    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    end.inference(df[:5])
