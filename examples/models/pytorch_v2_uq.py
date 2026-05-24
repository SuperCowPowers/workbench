"""PyTorch regression model trained with the V2 UQ pipeline (pure applicability domain).

Mirrors pytorch.py's regression flow but pins ``uq_version="v2"`` so the
endpoint emits V2 confidence — derived purely from the query's fingerprint
neighborhood (mean distance + neighbor target std). V0, V1, and V2 are all
fit + saved into the bundle; this hyperparameter just selects which one
drives the standard ``confidence`` / ``q_*`` columns. V2's q_* columns come
from the neighbor target distribution rather than being centered on the
model's prediction.
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Reuse the feature list and target from the original aqsol-regression model
source_model = Model("aqsol-regression")
feature_list = source_model.features()
target = source_model.target()

model_name = "aqsol-reg-pytorch-v2-uq"

# Recreate Flag in case you want to recreate the artifacts
recreate = True

# PyTorch Regression Model with V2 UQ
if recreate or not Model(model_name).exists():
    feature_set = FeatureSet("aqsol_features")
    m = feature_set.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.PYTORCH,
        feature_list=feature_list,
        target_column=target,
        description="PyTorch Regression Model for AQSol (V2 UQ — applicability domain)",
        tags=["pytorch", "molecular descriptors", "v2_uq"],
        hyperparameters={"max_epochs": 150, "layers": "128-64-32", "uq_version": "v2"},
    )
    m.set_owner("BW")

# Create an Endpoint for the Regression Model
if recreate or not Endpoint(model_name).exists():
    m = Model(model_name)
    end = m.to_endpoint(tags=["pytorch", "molecular descriptors", "v2_uq"])
    end.set_owner("BW")

    # Run inference on the endpoint
    end.auto_inference()
    end.cross_fold_inference()
