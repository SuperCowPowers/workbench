from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# Chemprop Regression Model
if recreate or not Model("aqsol-chemprop-reg").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-chemprop-reg",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility",
        description="Chemprop Regression Model for AQSol",
        tags=["chemprop", "aqsol"],
        # ChemProp hyperparameters: hidden_dim, depth, dropout, ffn_hidden_dim, ffn_num_layers, batch_size, max_epochs, patience
        hyperparameters={"max_epochs": 100, "hidden_dim": 300, "depth": 3},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("aqsol-chemprop-reg").exists():
    m = Model("aqsol-chemprop-reg")
    end = m.to_endpoint(tags=["chemprop", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)

# Chemprop Classification Model
if recreate or not Model("aqsol-chemprop-class").exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-chemprop-class",
        model_type=ModelType.CLASSIFIER,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility_class",
        description="Chemprop Classification Model for AQSol",
        tags=["chemprop", "aqsol"],
        hyperparameters={"max_epochs": 100, "hidden_dim": 300, "depth": 3},
    )
    m.set_owner("BW")
    m.set_class_labels(["low", "medium", "high"])

# Create an Endpoint for the Classification Model
if recreate or not Endpoint("aqsol-chemprop-class").exists():
    m = Model("aqsol-chemprop-class")
    end = m.to_endpoint(tags=["chemprop", "molecular descriptors"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)
