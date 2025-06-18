from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType, Endpoint

# Grab our existing AQSol Regression Model
model = Model("aqsol-regression")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Transform FeatureSet into Quantile Regression Model
fs = FeatureSet("aqsol_features")
my_model = fs.to_model(
    name="aqsol-quantiles",
    model_type=ModelType.UQ_REGRESSOR,
    target_column="solubility",
    feature_list=features,
    description="AQSol Quantile Regression",
    tags=["aqsol", "quantiles"],
)

# Deploy an Endpoint for the Model
end = my_model.to_endpoint(tags=["aqsol", "quantiles"])

# Run auto-inference on the AQSol Quantile Regression Endpoint
end.auto_inference(capture=True)
