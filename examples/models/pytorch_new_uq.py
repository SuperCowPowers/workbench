"""PyTorch regression model trained with the V1 UQ pipeline (proximity-augmented RF).

Mirrors pytorch.py's regression flow but pins ``uq_version="v1"`` so the
endpoint emits V1 confidence (the proximity + RandomForest residual estimator)
rather than the default V0 (isotonic-on-(prediction, std)). Both V0 and V1 are
still fit + saved into the bundle; this hyperparameter just selects which one
populates the standard ``confidence`` / ``q_*`` columns at inference.
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Reuse the feature list and target from the original aqsol-regression model
source_model = Model("aqsol-regression")
feature_list = source_model.features()
target = source_model.target()

model_name = "aqsol-reg-pytorch-new-uq"

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# PyTorch Regression Model with V1 UQ
if recreate or not Model(model_name).exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.PYTORCH,
        feature_list=feature_list,
        target_column=target,
        description="PyTorch Regression Model for AQSol (V1 UQ)",
        tags=["pytorch", "molecular descriptors", "new_uq"],
        hyperparameters={"max_epochs": 150, "layers": "128-64-32", "uq_version": "v1"},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint(model_name).exists():
    m = Model(model_name)
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors", "new_uq"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference()
    end.cross_fold_inference()
