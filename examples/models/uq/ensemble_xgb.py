from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.utils.model_utils import get_custom_script_path

# Grab our existing AQSol Regression Model
model = Model("aqsol-regression")

# Get the custom script path for the proximity model
script_path = get_custom_script_path("uq_models", "ensemble_xgb.template")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Transform FeatureSet into Bootstrap Ensemble (for UQ) Model
fs = FeatureSet("aqsol_features")
my_model = fs.to_model(
    name="aqsol-ensemble",
    model_type=ModelType.REGRESSOR,
    feature_list=features,
    target_column=target,
    description="AQSol Ensemble Regression (for UQ)",
    tags=["aqsol", "ensemble"],
    custom_script=script_path,
)

# Deploy an Endpoint for the Model
end = my_model.to_endpoint(tags=["aqsol", "ensemble", "uq"])

# Run auto-inference on the AQSol Ensemble/UQ Endpoint
end.auto_inference()
