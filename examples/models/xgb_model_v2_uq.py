"""XGBoost regression model trained with the V2 UQ pipeline (pure applicability domain).

Mirrors xgb_model.py's regression flow but pins ``uq_version="v2"`` so the
endpoint emits V2 confidence — derived purely from the query's fingerprint
neighborhood (mean distance + neighbor target std), no ensemble signal, no
RandomForest. V0, V1, and V2 are all fit + saved into the bundle regardless;
this hyperparameter just selects which one populates the standard
``confidence`` / ``q_*`` columns at inference. V2's q_* columns are derived
from the *neighbor target distribution* rather than centered on the model's
prediction — when those disagree, the gap is itself a cliff diagnostic.
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

feature_list = [
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
target = "solubility"
model_name = "aqsol-regression-v2-uq"

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# XGBoost Regression Model with V2 UQ
if recreate or not Model(model_name).exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        feature_list=feature_list,
        target_column=target,
        description="XGBoost Regression Model for AQSol (V2 UQ — applicability domain)",
        tags=["xgboost", "molecular descriptors", "v2_uq"],
        hyperparameters={"n_estimators": 200, "max_depth": 6, "uq_version": "v2"},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint(model_name).exists():
    m = Model(model_name)
    end = m.to_endpoint(tags=["xgboost", "molecular descriptors", "v2_uq"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference()
    end.cross_fold_inference()
