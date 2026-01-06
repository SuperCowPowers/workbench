from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Recreate Flag in case you want to recreate the artifacts
recreate = False

# =============================================================================
# Single-Target Chemprop Regression Model
# =============================================================================
if recreate or not Model("open-admet-chemprop-logd").exists():
    feature_set = FeatureSet("open_admet_all")
    m = feature_set.to_model(
        name="open-admet-chemprop-logd",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",  # Single target (string)
        feature_list=["smiles"],
        description="Single-target Chemprop Regression Model for LogD",
        tags=["chemprop", "open_admet"],
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("open-admet-chemprop-logd").exists():
    m = Model("open-admet-chemprop-logd")
    end = m.to_endpoint(tags=["chemprop", "open_admet"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()

# =============================================================================
# Hybrid ChemProp Model (SMILES + Molecular Descriptors)
# =============================================================================
# This example shows how to combine ChemProp's learned molecular representations
# with pre-computed molecular descriptors (RDKit/Mordred features).
# The extra features are concatenated with the MPNN output before the FFN.

# Top 10 SHAP features from LogD XGBoost model (provides complementary information to MPNN)
TOP_LOGD_SHAP_FEATURES = [
    "mollogp",
    "fr_halogen",
    "peoe_vsa6",
    "nbase",
    "peoe_vsa7",
    "peoe_vsa9",
    "peoe_vsa1",
    "mi",
    "bcut2d_mrlow",
    "slogp_vsa1",
]

if recreate or not Model("open-admet-chemprop-logd-hybrid").exists():
    feature_set = FeatureSet("open_admet_logd")

    # Hybrid mode: SMILES + top molecular descriptors
    # The template auto-detects extra features beyond the SMILES column
    hybrid_features = ["smiles"] + TOP_LOGD_SHAP_FEATURES

    m = feature_set.to_model(
        name="open-admet-chemprop-logd-hybrid",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",
        feature_list=hybrid_features,
        description="Hybrid ChemProp model combining D-MPNN with top SHAP molecular descriptors",
        tags=["chemprop", "open_admet", "hybrid"],
    )
    m.set_owner("BW")

# Create an Endpoint for the Hybrid Model
if recreate or not Endpoint("open-admet-chemprop-logd-hybrid").exists():
    m = Model("open-admet-chemprop-logd-hybrid")
    end = m.to_endpoint(tags=["chemprop", "open_admet", "hybrid"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()

# =============================================================================
# Chemprop Classification Model
# =============================================================================
if recreate or not Model("aqsol-class-chemprop").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-class-chemprop",
        model_type=ModelType.CLASSIFIER,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility_class",
        feature_list=["smiles"],
        description="Classification ChemProp model AQSol solubility classes",
        tags=["chemprop", "aqsol", "class"],
    )
    m.set_owner("BW")
    m.set_class_labels(["low", "medium", "high"])

# Create an Endpoint for the Multi-Task Regression Model
if recreate or not Endpoint("aqsol-class-chemprop").exists():
    m = Model("aqsol-class-chemprop")
    end = m.to_endpoint(tags=["chemprop", "aqsol", "class"])
    end.set_owner("BW")
    end.auto_inference()
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
    end.auto_inference()
    end.cross_fold_inference()
