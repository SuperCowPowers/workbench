# Description: Create XGBoost, PyTorch, and ChemProp models for all Open ADMET FeatureSets
from workbench.api import FeatureSet, Model, ModelType, ModelFramework
from workbench_bridges.api import ParameterStore

# Set to True to recreate models even if they already exist
RECREATE = False

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

FS_LIST = ["open_admet_logd"]  # For testing purposes, only process logD


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

    # 0. We've seen a bunch of 0.0 values (in the clint models) so we're going to give those 0 sample weight
    df = fs.pull_dataframe()
    if "clint" in fs_name:
        zero_ids = df.loc[df[target] == 0.0, fs.id_column].tolist()
        fs.set_sample_weights({id: 0.0 for id in zero_ids})
        print(f"Set {len(zero_ids)}/{len(df)} samples with target == 0.0 to weight 0")
    else:
        fs.set_sample_weights({})  # Clear any existing sample weights

    # High Target Gradients: Not used right now
    """
    print("\nComputing High Target Gradients for sample weights...")
    prox = fs.prox_model(target, rdkit_features)
    htg_df = prox.target_gradients(top_percent=5.0, min_delta=0.25)  # Log space targets
    htg_ids = htg_df[fs.id_column].tolist()
    print(f"HTG Top 5% (min_delta 0.25): {len(htg_ids)}")

    # Outliers: HTG points that aren't in the isolated compounds (top 25%)
    isolated_df = prox.isolated(top_percent=25)
    isolated_ids = isolated_df[fs.id_column].tolist()
    print(f"Isolated Compounds (top 20%): {len(isolated_df)}")
    htg_ids = [id for id in htg_ids if id not in set(isolated_ids)]
    print(f"Outliers (HTG not Isolated): {len(htg_ids)}")

    # Print out the neighbors of the outliers
    for id in htg_ids:
        neighbors = prox.neighbors(id)
        print(neighbors)

    # Set sample weights to 0.25 for outlier IDs
    sample_weights = {id: 0.25 for id in htg_ids}
    fs.set_sample_weights(sample_weights)
    """

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
        )
        xgb_model.set_owner("BW")
        end = xgb_model.to_endpoint(tags=["open_admet", base_name, "xgboost"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference()
        end.full_inference()
        end.cross_fold_inference()

    # Get feature importances from the XGBoost model
    xgb_model = Model(xgb_model_name)
    importances = xgb_model.shap_importance()
    non_zero_shap = [feat for feat, imp in importances if imp != 0.0]
    top_50_shap = non_zero_shap[:50]

    # 2. Create PyTorch model (all RDKit/Mordred features)
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
        )
        pytorch_model.set_owner("BW")
        end = pytorch_model.to_endpoint(tags=["open_admet", base_name, "pytorch"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference()
        end.full_inference()
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
        )
        chemprop_model.set_owner("BW")
        end = chemprop_model.to_endpoint(tags=["open_admet", base_name, "chemprop"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference()
        end.full_inference()
        end.cross_fold_inference()

    # 4. Create ChemProp Hybrid model (SMILES + Top 50 SHAP features)
    chemprop_hybrid_model_name = f"{short_name}-reg-chemprop-hybrid"
    if True or RECREATE or not Model(chemprop_hybrid_model_name).exists():
        print(f"Creating ChemProp Hybrid model: {chemprop_hybrid_model_name}")
        chemprop_hybrid_model = fs.to_model(
            name=chemprop_hybrid_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=target,
            feature_list=["smiles"] + top_50_shap,
            description=f"ChemProp D-MPNN Hybrid for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "chemprop", "hybrid"],
        )
        chemprop_hybrid_model.set_owner("BW")
        end = chemprop_hybrid_model.to_endpoint(tags=["open_admet", base_name, "chemprop", "hybrid"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference()
        end.full_inference()
        end.cross_fold_inference()

    # 4. Create an XGBoost Fingerprint Model
    fingerprint_model_name = f"{short_name}-reg-fp"
    if RECREATE or not Model(fingerprint_model_name).exists():
        print(f"Creating Fingerprint model: {fingerprint_model_name}")
        fingerprint_model = fs.to_model(
            name=fingerprint_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            target_column=target,
            feature_list=["fingerprint"],
            description=f"Fingerprint-based model for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "fingerprint"],
        )
        fingerprint_model.set_owner("BW")
        end = fingerprint_model.to_endpoint(tags=["open_admet", base_name, "fingerprint"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference()
        end.full_inference()
        end.cross_fold_inference()

    print(f"\nCompleted all models for: {fs_name}")

    # 5. Create a PyTorch Fingerprint Model
    pytorch_fp_model_name = f"{short_name}-reg-fp-pytorch"
    if RECREATE or not Model(pytorch_fp_model_name).exists():
        print(f"Creating PyTorch Fingerprint model: {pytorch_fp_model_name}")
        pytorch_fp_model = fs.to_model(
            name=pytorch_fp_model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.PYTORCH,
            target_column=target,
            feature_list=["fingerprint"],
            description=f"Fingerprint-based PyTorch model for {base_name} prediction",
            tags=["open_admet", base_name, "regression", "fingerprint", "pytorch"],
        )
        pytorch_fp_model.set_owner("BW")
        end = pytorch_fp_model.to_endpoint(tags=["open_admet", base_name, "fingerprint", "pytorch"], max_concurrency=1)
        end.set_owner("BW")
        end.auto_inference()
        end.full_inference()
        end.cross_fold_inference()


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
