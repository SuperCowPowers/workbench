from workbench.api import FeatureSet, Model, ModelType, Endpoint
from workbench.utils.model_utils import get_custom_script_path

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")
model = Model("aqsol-regression")
feature_list = model.features()
target = model.target()

# Recreate Flag in case you want to recreate the artifacts
recreate = False

# A 'Model' to Compute Molecular Descriptors Features
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

# Create an Endpoint for the Model
if recreate or not Endpoint("aqsol-pytorch-reg").exists():
    m = Model("aqsol-pytorch-reg")
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors"], serverless=False)
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference(capture=True)
