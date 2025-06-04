from pprint import pprint
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.utils.model_utils import get_custom_script_path

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")

# Grab our AQSol Regression Model
model = Model("aqsol-regression")

# Get the custom script path for the proximity model
script_path = get_custom_script_path("uq_models", "gaussian_process.template")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Create the GaussianProcess Model from our FeatureSet
fs = FeatureSet(model.get_input())
my_model = fs.to_model(
    name="aqsol-gaussian-reg",
    model_type=ModelType.REGRESSOR,
    feature_list=features,
    target_column=target,
    description=f"GaussianProcessRegressor Model for AQSol",
    tags=["aqsol", "gaussian", "regression"],
    custom_script=script_path,
)
# Print the details of the created model
pprint(my_model.details())
my_model = Model("aqsol-gaussian-reg")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(tags=["aqsol", "gaussian"])

# Run auto-inference on the Endpoint
endpoint.auto_inference(capture=True)
