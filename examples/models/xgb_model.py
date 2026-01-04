from workbench.api import FeatureSet, Model, ModelType, Endpoint

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")
model = Model("aqsol-regression")
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

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# XGBoost Regression Model
if recreate or not Model("aqsol-regression").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-regression",
        model_type=ModelType.UQ_REGRESSOR,
        feature_list=feature_list,
        target_column=target,
        description="XGBoost Regression Model for AQSol",
        tags=["xgboost", "molecular descriptors"],
        hyperparameters={"n_estimators": 200, "max_depth": 6},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("aqsol-regression").exists():
    m = Model("aqsol-regression")
    end = m.to_endpoint(tags=["xgboost", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference()
    end.cross_fold_inference()

# XGBoost Classification Model
if recreate or not Model("aqsol-class").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-class",
        model_type=ModelType.CLASSIFIER,
        feature_list=feature_list,
        target_column="solubility_class",
        description="XGBoost Classification Model for AQSol",
        tags=["xgboost", "molecular descriptors"],
    )
    m.set_owner("BW")
    m.set_class_labels(["low", "medium", "high"])

# Create an Endpoint for the Classification Model
if recreate or not Endpoint("aqsol-class").exists():
    m = Model("aqsol-class")
    end = m.to_endpoint(tags=["xgboost", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference()
    end.cross_fold_inference()
