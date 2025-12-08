# Description: Create a ChemProp model for CACO-2 Efflux Ratio prediction with exclusions.
# Prediction Target: CACO-2 ER
from workbench.api import FeatureSet, Model, ModelType, ModelFramework
from workbench_bridges.api import ParameterStore
from workbench.algorithms.dataframe.proximity import Proximity


if __name__ == "__main__":

    # Grab the FeatureSet
    fs = FeatureSet("open_admet_caco_2_efflux")
    df = fs.pull_dataframe()
    target = "caco_2_efflux"

    # Pull features from Parameter Store
    params = ParameterStore()
    features = params.get("/workbench/feature_lists/rdkit_mordred_stereo_v1")

    # Find "High Target Gradients" with a Proximity Model
    """ Note: Currently not used, but kept for reference
    df = fs.pull_dataframe()
    prox = Proximity(df, fs.id_column, features, target, track_columns=features)
    htg_df = prox.target_gradients(top_percent=10.0, min_delta=8.0)
    htg_ids = htg_df[fs.id_column].tolist()
    print(f"HTG Top 10% (min_delta 8.0): {len(htg_ids)}")

    # Print out the neighbors and their deltas
    show_cols = ["molecule_name", "neighbor_id", "distance", "caco_2_efflux"]
    for id in htg_ids:
        print(prox.neighbors(id)[show_cols])

    # Set sample weights to 0.0 for HTG compounds to exclude them from training
    fs.set_sample_weights({id: 0.0 for id in htg_ids})
    """

    # Create an XGBoost model with hyperparameter tuning
    hyperparameters = {
        # Core tree parameters
        "n_estimators": 200,  # More trees for better signal capture with ~320 features
        "max_depth": 6,  # Medium depth - you have good signal (top features with SHAP >1.0)
        "learning_rate": 0.05,  # Lower rate with more estimators for smoother learning
        # Sampling parameters
        "subsample": 0.7,  # Moderate row sampling to reduce overfitting
        "colsample_bytree": 0.6,  # More aggressive feature sampling given 320 features
        "colsample_bylevel": 0.8,  # Additional feature sampling at each tree level
        # Regularization
        "min_child_weight": 5,  # Higher to prevent overfitting on small groups
        "gamma": 0.2,  # Moderate pruning - you have real signal so don't over-prune
        "reg_alpha": 0.5,  # L1 for feature selection (useful with 320 features)
        "reg_lambda": 2.0,  # Strong L2 to smooth predictions
        # Random seed
        "random_state": 42,
    }
    ref_model = fs.to_model(
        name="caco2-efflux-reg-xgb-hp",
        model_type=ModelType.UQ_REGRESSOR,
        target_column=target,
        feature_list=features,
        description="XGBoost reference model with hyperparameter tuning for CACO-2 Efflux Ratio",
        tags=["caco2", "er", "regression", "xgboost", "reference", "hp"],
        hyperparameters=hyperparameters,
        train_all_data=True,
    )
    ref_model.set_owner("BW")
    end = ref_model.to_endpoint(tags=["caco2", "xgboost", "tuned"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()

    # Get feature importances from the reference model
    ref_model = Model("caco2-efflux-reg-hp")
    importances = ref_model.shap_importance()
    non_zero_shap = [feat for feat, imp in importances if imp != 0.0]
    top_50_features = non_zero_shap[:50]

    # Create a PyTorch Model
    model = fs.to_model(
        name="caco2-efflux-reg-pytorch",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.PYTORCH_TABULAR,
        target_column="caco_2_efflux",
        feature_list=non_zero_shap,
        description="PyTorch Tabular reference model for CACO-2 Efflux Ratio",
        tags=["caco2", "er", "regression", "pytorch", "reference"],
        hyperparameters={"n_folds": 10},
    )
    model.set_owner("BW")
    end = model.to_endpoint(tags=["caco2", "pytorch"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()

    # Create ChemProp Model
    hyperparameters = {
        "n_folds": 10,
        "hidden_dim": 300,
        "depth": 4,
        "dropout": 0.10,
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 2,
    }
    model = fs.to_model(
        name="caco2-efflux-reg-chemprop",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="caco_2_efflux",
        feature_list=["smiles"],  # Using only SMILES for ChemProp
        description="ChemProp D-MPNN for CACO-2 ER prediction",
        tags=["caco2", "er", "regression", "chemprop"],
        hyperparameters=hyperparameters,
    )
    model.set_owner("BW")

    # Create Endpoint and run inference
    end = model.to_endpoint(tags=["caco2", "chemprop"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()

    # Now create a hybrid ChemProp model with top 50 features
    features = ["smiles"] + top_50_features
    model = fs.to_model(
        name="caco2-efflux-reg-chemprop-hybrid",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="caco_2_efflux",
        feature_list=features,
        description="ChemProp D-MPNN Hybrid for CACO-2 ER prediction",
        tags=["caco2", "er", "regression", "chemprop", "hybrid"],
        hyperparameters=hyperparameters,
        train_all_data=True,
    )
    model.set_owner("BW")
    end = model.to_endpoint(tags=["caco2", "chemprop", "hybrid"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()

    # Read in the blind test data and run inference
    # test_df = pd.read_csv("test_data_blind.csv")
    # end.inference(test_df, capture_name="blind_test")
