import os
from pprint import pprint
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType

hyperparameters = {
    "n_estimators": 200,  # Number of trees
    "max_depth": 6,  # Maximum tree depth
    "learning_rate": 0.05,  # Boosting learning rate
    "subsample": 0.7,  # Row sampling ratio
    "colsample_bytree": 0.3,  # Feature sampling per tree
    "colsample_bylevel": 0.5,  # Feature sampling per level
    "min_child_weight": 5,  # Minimum samples in leaf
    "gamma": 0.2,  # Minimum loss reduction for split
    "reg_alpha": 0.5,  # L1 regularization
    "reg_lambda": 2.0,  # L2 regularization
    "scale_pos_weight": 1,  # Class weight balance
}

# Recreate Flag in case you want to recreate the artifacts
recreate = os.environ.get("RECREATE", "False").lower() == "true"

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")

# Grab our AQSol Regression Model
model = Model("aqsol-regression")

# Get Feature and Target Columns from the existing given Model
features = model.features()
target = model.target()

# Create the Mapie Model from our FeatureSet
fs = FeatureSet(model.get_input())
if recreate or not Model("aqsol-uq-hyper").exists():
    my_model = fs.to_model(
        name="aqsol-uq-hyper",
        model_type=ModelType.UQ_REGRESSOR,
        feature_list=features,
        target_column=target,
        description="Mapie + XGB Model",
        tags=["aqsol", "mapie", "regression"],
        hyperparameters=hyperparameters,
    )
    # Print the details of the created model
    pprint(my_model.details())
    my_model = Model("aqsol-uq-hyper")

    # Deploy an Endpoint for the Model
    endpoint = my_model.to_endpoint()

    # Run auto-inference on the Endpoint
    endpoint.auto_inference(capture=True)
    endpoint.cross_fold_inference()
