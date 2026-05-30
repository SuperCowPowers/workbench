"""ChemProp regression model trained with the V2 UQ pipeline (pure applicability domain).

Mirrors chemprop_new_uq.py's regression flow but pins ``uq_version="v2"`` so
the endpoint emits V2 confidence — derived purely from the query's
fingerprint neighborhood (mean distance + neighbor target std). V0, V1, and
V2 are all fit + saved into the bundle; this hyperparameter just selects
which one drives the standard ``confidence`` / ``q_*`` columns. V2's q_*
columns come from the neighbor target distribution rather than being centered
on the model's prediction — when those disagree, the gap is itself a cliff
diagnostic.
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint, PublicData

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# =============================================================================
# Chemprop Regression Model (V2 UQ — applicability domain)
# =============================================================================
model_name = "logd-reg-chemprop-v2-uq"
if recreate or not Model(model_name).exists():
    feature_set = FeatureSet("open_admet_logd")
    m = feature_set.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",  # Single target (string)
        feature_list=["smiles"],
        description="Single-target Chemprop Regression Model for LogD (V2 UQ)",
        tags=["chemprop", "open_admet", "v2_uq"],
        hyperparameters={"uq_version": "v2"},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint(model_name).exists():
    m = Model(model_name)
    end = m.to_endpoint(tags=["chemprop", "open_admet", "v2_uq"])
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()

    # Pull test data for inference
    pub_data = PublicData()
    df = pub_data.get("comp_chem/open_admet_expansionrx/test_logd")
    end.inference(df, capture_name="test_hold_out")
