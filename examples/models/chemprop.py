from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# =============================================================================
# Single-Target Chemprop Regression Model
# =============================================================================
if recreate or not Model("open-admet-chemprop-logd").exists():
    feature_set = FeatureSet("open_admet_all")
    m = feature_set.to_model(
        name="open-admet-chemprop-logd",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",  # Single target (string)
        feature_list=["smiles"],
        description="Single-target Chemprop Regression Model for LogD",
        tags=["chemprop", "open_admet"],
        hyperparameters={"max_epochs": 400, "hidden_dim": 300, "depth": 3, "n_folds": 5},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("open-admet-chemprop-logd").exists():
    m = Model("open-admet-chemprop-logd")
    end = m.to_endpoint(tags=["chemprop", "open_admet"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()

# =============================================================================
# Multi-Target Chemprop Regression Model (Open ADMET)
# =============================================================================
# This example shows how to train a multi-task model predicting all 9 ADMET endpoints
# from the Open ADMET challenge: logd, ksol, hlm_clint, mlm_clint, caco_2_papp_a_b,
# caco_2_efflux, mppb, mbpb, mgmb

ADMET_TARGETS = [
    "logd",
    "ksol",
    "hlm_clint",
    "mlm_clint",
    "caco_2_papp_a_b",
    "caco_2_efflux",
    "mppb",
    "mbpb",
    "mgmb",
]

if recreate or not Model("open-admet-chemprop-mt").exists():
    feature_set = FeatureSet("open_admet_all")
    m = feature_set.to_model(
        name="open-admet-chemprop-mt",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=ADMET_TARGETS,  # Multi-task: list of 9 targets
        feature_list=["smiles"],
        description="Multi-task ChemProp model for 9 ADMET endpoints",
        tags=["chemprop", "open_admet", "multitask"],
    )
    m.set_owner("BW")

# Create an Endpoint for the Multi-Task Regression Model
if recreate or not Endpoint("open-admet-chemprop-mt").exists():
    m = Model("open-admet-chemprop-mt")
    end = m.to_endpoint(tags=["chemprop", "open_admet", "multitask"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()
