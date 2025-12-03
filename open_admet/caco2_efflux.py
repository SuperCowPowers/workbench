# Description: Create a ChemProp model for CACO-2 Efflux Ratio prediction with exclusions.
# Prediction Target: CACO-2 ER
import pandas as pd
from workbench.api import FeatureSet, Model, ModelType, ModelFramework
from workbench_bridges.api import ParameterStore


if __name__ == "__main__":

    # Grab the FeatureSet
    fs = FeatureSet("open_admet_caco_2_efflux")
    df = fs.pull_dataframe()
    target = "caco_2_efflux"

    # Exclusions: We're not interesting an any compounds with ratios > 200
    range_cond = df[target] > 200
    excludes = df[range_cond][fs.id_column].tolist()
    fs.set_sample_weights({id: 0.0 for id in excludes})

    # Pull features from Parameter Store
    params = ParameterStore()
    features = params.get("/workbench/feature_lists/rdkit_mordred_stereo_v1")

    # Create an XGBoost reference model to get feature importances
    """
    ref_model = fs.to_model(
        name="caco2-efflux-reg-xgb",
        model_type=ModelType.UQ_REGRESSOR,
        target_column=target,
        feature_list=features,
        description="XGBoost reference model for CACO-2 Efflux Ratio",
        tags=["caco2", "er", "regression", "xgboost", "reference"],
        train_all_data=True,
    )
    ref_model.set_owner("BW")
    end = ref_model.to_endpoint(tags=["caco2", "xgboost"])
    end.set_owner("BW")
    end.auto_inference(capture=True)
    end.cross_fold_inference()
    """

    # Get feature importances from the reference model
    ref_model = Model("caco2-efflux-reg-xgb")
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
