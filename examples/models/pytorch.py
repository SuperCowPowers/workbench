from workbench.api import FeatureSet, Model, ModelType, Endpoint

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")
model = Model("aqsol-regression")
feature_list = model.features()
target = model.target()

# Recreate Flag in case you want to recreate the artifacts
recreate = False

# PyTorch Regression Model
if recreate or not Model("aqsol-pytorch-reg").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-pytorch-reg",
        model_type=ModelType.REGRESSOR,
        model_class="PyTorch",
        feature_list=feature_list,
        target_column=target,
        description="PyTorch Regression Model for AQSol",
        tags=["pytorch", "molecular descriptors"],
        hyperparameters={"layers": "256-512-512", "max_epochs": 150},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("aqsol-pytorch-reg").exists():
    m = Model("aqsol-pytorch-reg")
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)

# Pytorch Classification Model
if recreate or not Model("aqsol-pytorch-class").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-pytorch-class",
        model_type=ModelType.CLASSIFIER,
        model_class="PyTorch",
        feature_list=feature_list,
        target_column="solubility_class",
        description="PyTorch Classification Model for AQSol",
        tags=["pytorch", "molecular descriptors"],
        hyperparameters={"layers": "1024-512", "max_epochs": 50},
    )
    m.set_owner("BW")
    m.set_class_labels(["low", "medium", "high"])

# Create an Endpoint for the Classification Model
if recreate or not Endpoint("aqsol-pytorch-class").exists():
    m = Model("aqsol-pytorch-class")
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)
