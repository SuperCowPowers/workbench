from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# Chemprop Single-Target Regression Model
if recreate or not Model("aqsol-chemprop-reg").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-chemprop-reg",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=["solubility"],  # Now a list (single target)
        feature_list=["smiles"],  # Chemprop uses SMILES as input
        description="Chemprop Regression Model for AQSol",
        tags=["chemprop", "aqsol"],
        # ChemProp hyperparameters: hidden_dim, depth, dropout, ffn_hidden_dim, ffn_num_layers, batch_size, max_epochs, patience
        hyperparameters={"max_epochs": 400, "hidden_dim": 300, "depth": 3, "n_folds": 5},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("aqsol-chemprop-reg").exists():
    m = Model("aqsol-chemprop-reg")
    end = m.to_endpoint(tags=["chemprop", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)


# =============================================================================
# Multi-Target Chemprop Model (Open ADMET)
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
        name="open-admet-chemprop-mtp",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=ADMET_TARGETS,  # Multi-task: list of 9 targets
        feature_list=["smiles"],  # SMILES-only (no hybrid features)
        description="Multi-task ChemProp model for 9 ADMET endpoints",
        tags=["chemprop", "open_admet", "multitask"],
        hyperparameters={
            "max_epochs": 400,
            "hidden_dim": 300,
            "depth": 3,
            "n_folds": 5,
            "patience": 20,
        },
    )
    m.set_owner("BW")

# Create an Endpoint for the Multi-Task Model
if recreate or not Endpoint("open-admet-chemprop-mt").exists():
    m = Model("open-admet-chemprop-mtp")
    end = m.to_endpoint(tags=["chemprop", "open_admet", "multitask"])
    end.set_owner("BW")

    # Run inference - will output {target}_pred and {target}_pred_std for each target
    end.auto_inference(capture=True)
