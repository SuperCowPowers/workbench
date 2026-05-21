from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint, PublicData

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# =============================================================================
# Chemprop Regression Model (with new UQ)
# =============================================================================
model_name = "logd-reg-chemprop-new-uq"
if recreate or not Model(model_name).exists():
    feature_set = FeatureSet("open_admet_logd")
    m = feature_set.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",  # Single target (string)
        feature_list=["smiles"],
        description="Single-target Chemprop Regression Model for LogD",
        tags=["chemprop", "open_admet"],
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint(model_name).exists():
    m = Model(model_name)
    end = m.to_endpoint(tags=["chemprop", "open_admet"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()

    # Pull test data for inference
    pub_data = PublicData()
    df = pub_data.get("comp_chem/open_admet_expansionrx/test_logd")
    end.inference(df, capture_name="test_hold_out")
