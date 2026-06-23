"""Create the AQSol solubility classification Model and Endpoint (variant 2).

A copy of aqsol_class.py with different XGBoost hyperparameters -- a second
challenger for the same target (more, shallower trees at a higher learning rate).

Models:
    - aqsol-class-2
Endpoints:
    - aqsol-class-2
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework

FEATURE_LIST = [
    "molwt",
    "mollogp",
    "molmr",
    "heavyatomcount",
    "numhacceptors",
    "numhdonors",
    "numheteroatoms",
    "numrotatablebonds",
    "numvalenceelectrons",
    "numaromaticrings",
    "numsaturatedrings",
    "numaliphaticrings",
    "ringcount",
    "tpsa",
    "labuteasa",
    "balabanj",
    "bertzct",
]

# Differs from aqsol_class.py (defaults: 300 / 7 / 0.05)
HYPERPARAMETERS = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.08}

if __name__ == "__main__":

    # Build the classification Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-class-2",
        model_type=ModelType.CLASSIFIER,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility_class",
        feature_list=FEATURE_LIST,
        description="AQSol Classification Model (variant 2)",
        tags=["aqsol", "classification"],
        hyperparameters=HYPERPARAMETERS,
    )
    m.set_owner("test")
    m.set_class_labels(["low", "medium", "high"])

    # Deploy the classification Endpoint
    m = Model("aqsol-class-2")
    end = m.to_endpoint(tags=["aqsol", "classification"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
