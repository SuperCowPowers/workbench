"""Single-task CYP3A4 Chemprop on Octant data — the ST-RAE plumbing check.

Octant is the only public CYP source carrying credible intervals, so it is the only
way to exercise the challenge's actual metric end-to-end before the real data lands.
`compute_regression_metrics` adds an `st_rae` column when the target has
`<target>_ci_lower` / `_ci_upper` alongside it, so the question this script answers is
whether those columns survive FeatureSet creation, training, and inference capture to
reach the metrics computation on a deployed endpoint.

Small (1,084 curves, all QC PASS) — this is about the path, not the score.

Build the FeatureSet first: python cyp_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_octant"
MODEL_NAME = "cyp3a4-reg-chemprop-octant"
TARGET = "cyp3a4_pic50"
TAGS = ["openadmet_cyp", "chemprop", "activity", "credible_intervals"]

fs = FeatureSet(FS_NAME)

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=TARGET,
    description="CYP3A4 Chemprop on Octant dose-response, carries credible intervals for ST-RAE",
    tags=TAGS,
    hyperparameters={"uq_version": "v1"},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()

# The point of this script: confirm st_rae actually reaches the model's metrics.
metrics = model.get_inference_metrics()
if metrics is not None and "st_rae" in metrics.columns:
    print(f"st_rae present in inference metrics: {metrics['st_rae'].iloc[0]:.3f}")
else:
    cols = None if metrics is None else metrics.columns.tolist()
    print(f"st_rae MISSING from inference metrics (columns: {cols}) — credible intervals did not survive")
