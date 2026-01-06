from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint


def test_pytorch_models():
    # Grab our features and target
    model = Model("aqsol-regression")
    feature_list = model.features()
    target = model.target()

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # PyTorch Regression Model
    if recreate or not Model("aqsol-reg-pytorch").exists():
        feature_set = FeatureSet("aqsol_features")
        m = feature_set.to_model(
            name="aqsol-reg-pytorch",
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.PYTORCH,
            feature_list=feature_list,
            target_column=target,
            description="PyTorch Regression Model for AQSol",
            tags=["pytorch", "molecular descriptors"],
            hyperparameters={"max_epochs": 150, "layers": "128-64-32"},
        )
        m.set_owner("BW")

    # Create an Endpoint for the Regression Model
    if recreate or not Endpoint("aqsol-reg-pytorch").exists():
        m = Model("aqsol-reg-pytorch")
        end = m.to_endpoint(tags=["pytorch", "molecular descriptors"])
        end.set_owner("BW")

        # Run inference on the endpoint
        end.auto_inference()

    # Pytorch Classification Model
    if recreate or not Model("aqsol-class-pytorch").exists():
        feature_set = FeatureSet("aqsol_features")
        m = feature_set.to_model(
            name="aqsol-class-pytorch",
            model_type=ModelType.CLASSIFIER,
            model_framework=ModelFramework.PYTORCH,
            feature_list=feature_list,
            target_column="solubility_class",
            description="PyTorch Classification Model for AQSol",
            tags=["pytorch", "molecular descriptors"],
        )
        m.set_owner("BW")
        m.set_class_labels(["low", "medium", "high"])

    # Create an Endpoint for the Classification Model
    if recreate or not Endpoint("aqsol-class-pytorch").exists():
        m = Model("aqsol-class-pytorch")
        end = m.to_endpoint(tags=["pytorch", "molecular descriptors"])
        end.set_owner("BW")

        # Run inference on the endpoint
        end.auto_inference()


if __name__ == "__main__":
    test_pytorch_models()
