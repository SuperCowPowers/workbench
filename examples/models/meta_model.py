from workbench.api import MetaModel, Endpoint

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# =============================================================================
# Meta Model: Ensemble of XGBoost, PyTorch, and ChemProp for LogD
# =============================================================================
# This meta model aggregates predictions from different model architectures,
# using model-weighted voting to combine their outputs.

# Child endpoints that this meta model will call
CHILD_ENDPOINTS = [
    "logd-reg-pytorch",
    "logd-reg-chemprop",
]

if recreate or not MetaModel("logd-meta").exists():
    m = MetaModel.create(
        name="logd-meta",
        child_endpoints=CHILD_ENDPOINTS,
        target_column="logd",
        description="Meta model combining XGBoost, PyTorch, and ChemProp for LogD prediction",
        tags=["meta", "open_admet", "logd"],
    )
    m.set_owner("BW")

# Create an Endpoint for the Meta Model
if recreate or not Endpoint("logd-meta").exists():
    m = MetaModel("logd-meta")
    end = m.to_endpoint(tags=["meta", "open_admet", "logd"])
    end.set_owner("BW")
    end.auto_inference()
    end.full_inference()
