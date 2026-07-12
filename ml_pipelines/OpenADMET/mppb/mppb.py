"""Create regression models (XGBoost, PyTorch, ChemProp, ChemProp-Hybrid) for the
Open ADMET mppb assay.

Consumes the shared FeatureSet built by ../load_data/load_data.py and produces a model +
endpoint per framework. SHAP importances from the XGBoost model drive the PyTorch and
ChemProp-Hybrid feature lists.
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

FS_NAME = "open_admet_mppb"
RECREATE = True


def main():
    # RDKit/Mordred feature columns (the tabular feature space for xgb/pytorch)
    rdkit_features = Endpoint("smiles-to-2d-v1").output_columns()

    fs = FeatureSet(FS_NAME)
    target = FS_NAME.replace("open_admet_", "")
    short_name = target.replace("_", "-")

    # 1. XGBoost (RDKit/Mordred features)
    xgb_name = f"{short_name}-reg-xgb"
    if RECREATE or not Model(xgb_name).exists():
        print(f"Creating XGBoost model: {xgb_name}")
        model = fs.to_model(
            name=xgb_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column=target,
            feature_list=rdkit_features,
            description=f"XGBoost model for {target} prediction",
            tags=["open_admet", target, "regression", "xgboost"],
        )
        model.set_owner("BW")
        end = model.to_endpoint(tags=["open_admet", target, "xgboost"], max_concurrency=1)
        end.set_owner("BW")
        end.test_inference()
        end.cross_fold_inference()

    # SHAP importances from the XGBoost model drive the pytorch/hybrid feature lists
    importances = Model(xgb_name).shap_importance()
    non_zero_shap = [feat for feat, imp in importances if imp != 0.0]
    top_50_shap = non_zero_shap[:50]

    # 2. PyTorch (non-zero SHAP features)
    pytorch_name = f"{short_name}-reg-pytorch"
    if RECREATE or not Model(pytorch_name).exists():
        print(f"Creating PyTorch model: {pytorch_name}")
        model = fs.to_model(
            name=pytorch_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.PYTORCH,
            target_column=target,
            feature_list=non_zero_shap,
            description=f"PyTorch Tabular model for {target} prediction",
            tags=["open_admet", target, "regression", "pytorch"],
        )
        model.set_owner("BW")
        end = model.to_endpoint(tags=["open_admet", target, "pytorch"], max_concurrency=1)
        end.set_owner("BW")
        end.test_inference()
        end.cross_fold_inference()

    # 3. ChemProp (SMILES only)
    chemprop_name = f"{short_name}-reg-chemprop"
    if RECREATE or not Model(chemprop_name).exists():
        print(f"Creating ChemProp model: {chemprop_name}")
        model = fs.to_model(
            name=chemprop_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=target,
            feature_list=["smiles"],
            description=f"ChemProp D-MPNN for {target} prediction",
            tags=["open_admet", target, "regression", "chemprop"],
        )
        model.set_owner("BW")
        end = model.to_endpoint(tags=["open_admet", target, "chemprop"], max_concurrency=1)
        end.set_owner("BW")
        end.test_inference()
        end.cross_fold_inference()

    # 4. ChemProp Hybrid (SMILES + top-50 SHAP features)
    hybrid_name = f"{short_name}-reg-chemprop-hybrid"
    if RECREATE or not Model(hybrid_name).exists():
        print(f"Creating ChemProp Hybrid model: {hybrid_name}")
        model = fs.to_model(
            name=hybrid_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=target,
            feature_list=["smiles"] + top_50_shap,
            description=f"ChemProp D-MPNN Hybrid for {target} prediction",
            tags=["open_admet", target, "regression", "chemprop", "hybrid"],
        )
        model.set_owner("BW")
        end = model.to_endpoint(tags=["open_admet", target, "chemprop", "hybrid"], max_concurrency=1)
        end.set_owner("BW")
        end.test_inference()
        end.cross_fold_inference()


if __name__ == "__main__":
    main()
