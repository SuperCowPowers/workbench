from pprint import pprint
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.utils.model_utils import get_custom_script_path

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")

# Grab our AQSol Regression Model
model = Model("aqsol-regression")

# Get the custom script path for the proximity model
script_path = get_custom_script_path("uq_models", "mapie_xgb.template")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Create the Mapie XGB Model from our FeatureSet
fs = FeatureSet(model.get_input())
my_model = fs.to_model(
    name="aqsol-mapie-reg",
    model_type=ModelType.REGRESSOR,
    feature_list=features,
    target_column=target,
    description=f"Mapie XGB Model for AQSol",
    tags=["aqsol", "mapie", "regression"],
    custom_script=script_path,
)
# Print the details of the created model
pprint(my_model.details())
my_model = Model("aqsol-mapie-reg")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(tags=["aqsol", "mapie"])

# Run auto-inference on the Endpoint
endpoint.auto_inference(capture=True)
