from pprint import pprint
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.utils.model_utils import get_custom_script_path

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")

# Grab our AQSol Regression Model
model = Model("aqsol-regression")

# Get the custom script path for the UQ model
script_path = get_custom_script_path("uq_models", "ngboost.template")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Create the NGBoost Model from our FeatureSet
fs = FeatureSet(model.get_input())
my_model = fs.to_model(
    name="aqsol-ngboost-reg",
    model_type=ModelType.UQ_REGRESSOR,
    feature_list=features,
    target_column=target,
    description="NGBoost Model for AQSol",
    tags=["aqsol", "ngboost", "regression"],
    custom_script=script_path,
)
# Print the details of the created model
pprint(my_model.details())
my_model = Model("aqsol-ngboost-reg")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(tags=["aqsol", "ngboost"])

# Run auto-inference on the Endpoint
endpoint.auto_inference()
