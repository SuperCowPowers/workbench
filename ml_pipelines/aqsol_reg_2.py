"""Create the AQSol solubility regression Model and Endpoint (variant 2).

A copy of aqsol_reg.py with different XGBoost hyperparameters -- a second
challenger for the same target (more, shallower trees at a lower learning rate).

Models:
    - aqsol-regression-2
Endpoints:
    - aqsol-regression-2
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

# Differs from aqsol_reg.py (defaults: 300 / 7 / 0.05)
HYPERPARAMETERS = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03}

if __name__ == "__main__":

    # Build the regression Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-regression-2",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility",
        feature_list=FEATURE_LIST,
        description="AQSol Regression Model (variant 2)",
        tags=["aqsol", "regression"],
        hyperparameters=HYPERPARAMETERS,
    )
    m.set_owner("test")

    # Deploy the regression Endpoint
    m = Model("aqsol-regression-2")
    end = m.to_endpoint(tags=["aqsol", "regression"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
