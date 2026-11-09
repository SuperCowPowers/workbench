"""Target transform: train in log10 space, serve predictions on the original scale.

MPPB is percent unbound -- physically bounded [0, 100], heavily skewed toward zero, and the
range that matters clinically is under 1%. A log10 transform spends model resolution on that
low tail instead of on the handful of high-unbound compounds.

The offset is the assay's LLOQ (0.1% here), so rows recorded as 0 land at -1 in transformed
space rather than failing log10's domain check.

The mppb_raw FeatureSet is built by ml_pipelines/Testing/target_transform/target_transform_fs.py
"""

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType

NAME = "mppb-log10-example"

# Recreate Flag in case you want to recreate the artifacts
recreate = False

fs = FeatureSet("mppb_raw")
feature_list = Endpoint("smiles-to-2d-v1").output_columns()

if recreate or not Model(NAME).exists():
    model = fs.to_model(
        name=NAME,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="mppb",
        feature_list=feature_list,
        description="MPPB percent unbound, log10 target transform",
        tags=["mppb", "target_transform"],
        hyperparameters={
            "target_transform": {"name": "log10", "offset": 0.1},
            "prediction_bounds": [0.0, 100.0],
        },
    )
    model.set_owner("BW")

if recreate or not Endpoint(NAME).exists():
    end = Model(NAME).to_endpoint(tags=["mppb", "target_transform"])
    end.set_owner("BW")
    end.cross_fold_inference()

# Predictions come back in percent unbound, and so does their spread
end = Endpoint(NAME)
predictions = end.inference(fs.pull_dataframe().head(10))
print(predictions[["mppb", "prediction", "prediction_std"]])

# Metrics are computed on the original scale, so this model compares directly against one
# trained without a transform
model = Model(NAME)
print(model.get_inference_metrics("full_cross_fold"))
