from workbench.api import FeatureSet, Model, ModelType, Endpoint
from workbench.utils.model_utils import get_custom_script_path

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")
model = Model("aqsol-regression")
feature_list = model.features()
target = model.target()

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# PyTorch Regression Model
if recreate or not Model("aqsol-pytorch-reg").exists():
    script_path = get_custom_script_path("pytorch_models", "pytorch.template")
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-pytorch-reg",
        model_type=ModelType.REGRESSOR,
        feature_list=feature_list,
        target_column=target,
        description="PyTorch Regression Model for AQSol",
        tags=["pytorch", "molecular descriptors"],
        custom_script=script_path,
        training_image="pytorch_training",
        inference_image="pytorch_inference",
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint("aqsol-pytorch-reg").exists():
    m = Model("aqsol-pytorch-reg")
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors"], serverless=False)
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)

# Pytorch Classification Model
if recreate or not Model("aqsol-pytorch-class").exists():
    script_path = get_custom_script_path("pytorch_models", "pytorch.template")
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name="aqsol-pytorch-class",
        model_type=ModelType.CLASSIFIER,
        feature_list=feature_list,
        target_column="solubility_class",
        description="PyTorch Classification Model for AQSol",
        tags=["pytorch", "molecular descriptors"],
        custom_script=script_path,
        training_image="pytorch_training",
        inference_image="pytorch_inference",
    )
    m.set_owner("BW")

# Create an Endpoint for the Classification Model
if recreate or not Endpoint("aqsol-pytorch-class").exists():
    m = Model("aqsol-pytorch-class")
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors"], serverless=False)
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)
