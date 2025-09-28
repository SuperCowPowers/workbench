"""This Script creates the Workbench Artifacts in AWS needed for the tests

Models:
    - abalone-regression-bad
    - abalone-regression-good
Endpoints:
    - abalone-regression-bad
    - abalone-regression-good
"""

from workbench.api import FeatureSet, Model, ModelType

recreate = False

features = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
    "sex",
]

# Create the 'bad' Model and Endpoint
if recreate or not Model("abalone-regression-bad").exists():
    fs = FeatureSet("abalone_features")
    m = fs.to_model(
        name="abalone-regression-bad",
        model_type=ModelType.REGRESSOR,
        feature_list=features,
        target_column="class_number_of_rings",
        hyperparameters={
            "n_estimators": 1,  # Just one tree!
            "max_depth": 1,  # Basically a stump
            "learning_rate": 0.01,  # Learn super slowly (not that it matters with 1 tree)
            "subsample": 0.1,  # Only look at 10% of data
            "colsample_bytree": 0.1,  # Only use 10% of features
        },
    )
    m.set_owner("test")
    end = m.to_endpoint()
    end.auto_inference(capture=True)
    end.cross_fold_inference()

# Create the 'good' Model and Endpoint
if recreate or not Model("abalone-regression-good").exists():
    fs = FeatureSet("abalone_features")
    m = fs.to_model(
        name="abalone-regression-good",
        model_type=ModelType.REGRESSOR,
        feature_list=features,
        target_column="class_number_of_rings",
        hyperparameters={
            "n_estimators": 100,  # Reasonable for small dataset
            "max_depth": 4,  # Shallow to avoid overfitting
            "learning_rate": 0.1,  # Standard learning rate
            "subsample": 0.8,  # Some row sampling
            "colsample_bytree": 0.8,  # Some feature sampling
            "min_child_weight": 3,  # Regularization for small dataset
            "gamma": 0.1,  # Slight pruning
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
        },
    )
    m.set_owner("test")
    end = m.to_endpoint()
    end.auto_inference(capture=True)
    end.cross_fold_inference()
