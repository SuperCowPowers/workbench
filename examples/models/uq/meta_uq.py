from pprint import pprint
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.utils.model_utils import get_custom_script_path

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")

# Grab our AQSol Regression Model
model = Model("aqsol-regression")

# Get the custom script path for the UQ model
script_path = get_custom_script_path("uq_models", "meta_uq.template")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Create the NGBoost Model from our FeatureSet
fs = FeatureSet(model.get_input())
my_model = fs.to_model(
    name="aqsol-meta-uq",
    model_type=ModelType.UQ_REGRESSOR,
    feature_list=features,
    target_column=target,
    description="Meta UQ Model for AQSol",
    tags=["aqsol", "meta-uq", "regression"],
    custom_script=script_path,
    custom_args={"id_column": "id", "track_columns": "solubility"},
)
# Print the details of the created model
pprint(my_model.details())
my_model = Model("aqsol-meta-uq")

# Deploy an Endpoint for the Model
endpoint = my_model.to_endpoint(tags=["aqsol", "meta-uq"])

# Run auto-inference on the Endpoint
endpoint.auto_inference()
