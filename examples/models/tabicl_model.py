from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")
model = Model("aqsol-regression")
feature_list = model.features()
target = model.target()

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# TabICL Regression Model (a tabular foundation model: in-context learning, no gradient training)
if recreate or not Model("aqsol-reg-tabicl").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-reg-tabicl",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.TABICL,
        feature_list=feature_list,
        target_column=target,
        description="TabICL Regression Model for AQSol",
        tags=["tabicl", "molecular descriptors"],
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model. Serverless is refused when the model's
# measured serving memory is over 5 GB (this one is ~9 GB), so deploy real-time: the
# instance is sized from the measured memory.
if recreate or not Endpoint("aqsol-reg-tabicl").exists():
    m = Model("aqsol-reg-tabicl")
    end = m.to_endpoint(serverless=False, tags=["tabicl", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.test_inference()
    end.cross_fold_inference()
