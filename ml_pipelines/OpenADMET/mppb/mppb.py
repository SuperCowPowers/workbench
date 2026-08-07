"""Create regression models (XGBoost, PyTorch, ChemProp, ChemProp-Hybrid) for the
Open ADMET mppb assay.

Consumes the shared FeatureSet built by ../load_data/load_data.py and produces a model +
endpoint per framework. SHAP importances from the XGBoost model drive the PyTorch and
ChemProp-Hybrid feature lists.

Each model is scored three ways:

  - test_inference()          smoke test, not an evaluation
  - cross_fold_inference()    out-of-fold metrics from training time
  - inference(capture_name="rx_test_mppb")  the held-out ExpansionRx test rows

The test rows are never in the FeatureSet (see ../load_data/load_data.py), so the
rx_test_* capture is a genuine held-out score -- the models never trained on, validated
against, or calibrated UQ from those molecules. They arrive pre-featurized and
pre-transformed in the DFStore, so scoring is a straight inference pass.
"""

from workbench.api import DFStore, Endpoint, FeatureSet, Model, ModelType, ModelFramework

FS_NAME = "open_admet_mppb"

# Featurized + log-transformed ExpansionRx test set, written by ../load_data/load_data.py
TEST_STORE_KEY = "/workbench/datasets/open_admet_rx_test_featurized"


def capture_test_inference(end: Endpoint, target: str, features: list) -> None:
    """Score the held-out ExpansionRx test rows for this assay and persist the run."""
    test_df = DFStore().get(TEST_STORE_KEY).dropna(subset=[target])
    columns = ["molecule_name", "smiles", target] + features
    print(f"Running held-out test inference for {end.name} on {len(test_df)} rows")
    end.inference(test_df[columns], capture_name=f"rx_test_{target}", id_column="molecule_name")


def main():
    # RDKit/Mordred feature columns (the tabular feature space for xgb/pytorch)
    rdkit_features = Endpoint("smiles-to-2d-v1").output_columns()

    fs = FeatureSet(FS_NAME)
    target = FS_NAME.replace("open_admet_", "")
    short_name = target.replace("_", "-")

    # 1. XGBoost (RDKit/Mordred features)
    xgb_name = f"{short_name}-reg-xgb"
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
    capture_test_inference(end, target, rdkit_features)

    # SHAP importances from the XGBoost model drive the pytorch/hybrid feature lists
    importances = Model(xgb_name).shap_importance()
    non_zero_shap = [feat for feat, imp in importances if imp != 0.0]
    top_50_shap = non_zero_shap[:50]

    # 2. PyTorch (non-zero SHAP features)
    pytorch_name = f"{short_name}-reg-pytorch"
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
    capture_test_inference(end, target, non_zero_shap)

    # 3. ChemProp (SMILES only)
    chemprop_name = f"{short_name}-reg-chemprop"
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
    capture_test_inference(end, target, [])

    # 4. ChemProp Hybrid (SMILES + top-50 SHAP features)
    hybrid_name = f"{short_name}-reg-chemprop-hybrid"
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
    capture_test_inference(end, target, top_50_shap)


if __name__ == "__main__":
    main()
