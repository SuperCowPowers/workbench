# Description: Create XGBoost, PyTorch, and ChemProp models for all Open ADMET FeatureSets
from workbench.api import FeatureSet, Model, ModelType, ModelFramework
from workbench_bridges.api import ParameterStore

# Set to True to recreate models even if they already exist
RECREATE = True

# FeatureSet List
FS_LIST = [
    "open_admet_caco_2_efflux",
    "open_admet_caco_2_papp_a_b",
    "open_admet_hlm_clint",
    "open_admet_ksol",
    "open_admet_logd",
    "open_admet_mbpb",
    "open_admet_mgmb",
    "open_admet_mlm_clint",
    "open_admet_mppb",
]

# XGBoost hyperparameters
XGB_HYPERPARAMETERS = {
    # Objective: optimize for MAE
    "objective": "reg:absoluteerror",
    # Core tree parameters
    "n_estimators": 300,  # More trees for better convergence
    "max_depth": 6,
    "learning_rate": 0.03,  # Lower rate with more estimators
    # Sampling parameters
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 0.8,
    # Regularization
    "min_child_weight": 5,
    "gamma": 0.1,  # Slightly less aggressive pruning
    "reg_alpha": 0.3,  # L1 regularization
    "reg_lambda": 1.5,  # L2 regularization
    # Random seed
    "random_state": 42,
}

# PyTorch hyperparameters
PYTORCH_HYPERPARAMETERS = {
    "n_folds": 5,
}

# ChemProp hyperparameters
CHEMPROP_HYPERPARAMETERS = {
    "n_folds": 5,
    "hidden_dim": 300,
    "depth": 4,
    "dropout": 0.10,
    "ffn_hidden_dim": 300,
    "ffn_num_layers": 2,
}


def create_models_for_featureset(fs_name: str, rdkit_features: list[str]):
    """Create XGBoost, PyTorch, and ChemProp models for a given FeatureSet."""

    # Load the FeatureSet and get target column
    fs = FeatureSet(fs_name)

    # Target is just the last part of the FeatureSet name (after "open_admet_")
    target = fs_name.replace("open_admet_", "")

    # Derive base name from featureset name (remove "open_admet_" prefix)
    base_name = fs_name.replace("open_admet_", "")
    short_name = base_name.replace("_", "-")

    print(f"\n{'='*60}")
    print(f"Processing FeatureSet: {fs_name}")
    print(f"Target column: {target}")
    print(f"Base name: {short_name}")
    print(f"{'='*60}")

    # 1. Create XGBoost model
    xgb_model_name = f"{short_name}-reg-xgb"
    if RECREATE or not Model(xgb_model_name).exists():
        print(f"\nCreating XGBoost model: {xgb_model_name}")
        xgb_model = fs.to_model(
            name=xgb_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            target_column=target,
            feature_list=rdkit_features,
            description=f"XGBoost model for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "xgboost"],
            hyperparameters=XGB_HYPERPARAMETERS,
        )
        xgb_model.set_owner("BW")
        end = xgb_model.to_endpoint(tags=["open_admet", base_name, "xgboost"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference(capture=True)
        end.cross_fold_inference()

    # Get feature importances from the XGBoost model for PyTorch and hybrid models
    xgb_model = Model(xgb_model_name)
    importances = xgb_model.shap_importance()
    non_zero_shap = [feat for feat, imp in importances if imp != 0.0]
    top_50_features = non_zero_shap[:50]

    # 2. Create PyTorch model using non-zero SHAP features
    pytorch_model_name = f"{short_name}-reg-pytorch"
    if RECREATE or not Model(pytorch_model_name).exists():
        print(f"Creating PyTorch model: {pytorch_model_name}")
        pytorch_model = fs.to_model(
            name=pytorch_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.PYTORCH,
            target_column=target,
            feature_list=non_zero_shap,
            description=f"PyTorch Tabular model for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "pytorch"],
            hyperparameters=PYTORCH_HYPERPARAMETERS,
        )
        pytorch_model.set_owner("BW")
        end = pytorch_model.to_endpoint(tags=["open_admet", base_name, "pytorch"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference(capture=True)
        end.cross_fold_inference()

    # 3. Create ChemProp model (SMILES only)
    chemprop_model_name = f"{short_name}-reg-chemprop"
    if RECREATE or not Model(chemprop_model_name).exists():
        print(f"Creating ChemProp model: {chemprop_model_name}")
        chemprop_model = fs.to_model(
            name=chemprop_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=target,
            feature_list=["smiles"],
            description=f"ChemProp D-MPNN for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "chemprop"],
            hyperparameters=CHEMPROP_HYPERPARAMETERS,
        )
        chemprop_model.set_owner("BW")
        end = chemprop_model.to_endpoint(tags=["open_admet", base_name, "chemprop"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference(capture=True)
        end.cross_fold_inference()

    # 4. Create hybrid ChemProp model with top 50 features
    hybrid_model_name = f"{short_name}-reg-chemprop-hybrid"
    if RECREATE or not Model(hybrid_model_name).exists():
        print(f"Creating ChemProp Hybrid model: {hybrid_model_name}")
        hybrid_features = ["smiles"] + top_50_features
        hybrid_model = fs.to_model(
            name=hybrid_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=target,
            feature_list=hybrid_features,
            description=f"ChemProp D-MPNN Hybrid for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "chemprop", "hybrid"],
            hyperparameters=CHEMPROP_HYPERPARAMETERS,
        )
        hybrid_model.set_owner("BW")
        end = hybrid_model.to_endpoint(tags=["open_admet", base_name, "chemprop", "hybrid"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference(capture=True)
        end.cross_fold_inference()

    print(f"\nCompleted all models for: {fs_name}")


if __name__ == "__main__":
    # Pull features from Parameter Store
    params = ParameterStore()
    rdkit_features = params.get("/workbench/feature_lists/rdkit_mordred_stereo_v1")

    print(f"Processing {len(FS_LIST)} FeatureSets")
    print(f"Using {len(rdkit_features)} RDKit/Mordred features")

    # Loop over all FeatureSets and create models
    for fs_name in FS_LIST:
        create_models_for_featureset(fs_name, rdkit_features)

    print("\n" + "=" * 60)
    print("All models created successfully!")
    print("=" * 60)
