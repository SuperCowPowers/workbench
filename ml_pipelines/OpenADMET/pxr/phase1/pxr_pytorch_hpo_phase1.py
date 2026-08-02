"""PXR phase-1 model: hyperparameter-searched PyTorch tabular on the shared FeatureSet.

Same held-out setup as the other phase-1 models (the phase1_test rows are held out of
training via validation_ids and captured), on the 2D descriptor columns. The search runs
inside the single training job — trials are ephemeral, so only the winning config is
published as this model.

The search objective is `cv_mae` on scaffold folds of the *training* rows — the default,
and the one that matters here. Setting hpo["metric"]="holdout_mae" would tune the model on
Analog Set 1 and make the pxr_phase1_test capture optimistic, and unfair against the other
phase-1 models, which never see those labels during fitting.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

import logging

from workbench.api import Endpoint, FeatureSet, ModelFramework, ModelType
from workbench.training.hpo_harness import SearchSpace

log = logging.getLogger("workbench")

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-pytorch-hpo-phase1"
tags = ["openadmet_pxr", "pytorch", "hpo", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

# 2D descriptor columns. The FeatureSet also carries a 3D layer; leaving it out keeps this
# a straight read on hyperparameter search rather than a featurization comparison.
features = [c for c in Endpoint("smiles-to-2d-v1").output_columns() if c in df.columns]
log.info(f"=== {model_name} — {fs_name} ({len(features)} features, {len(phase1)} held-out rows) ===")

# The knobs to search — PyTorch's shipped space, unchanged:
#
#   layers         Choice(["128-64", "256-128", "512-256", "512-256-128", "1024-512-256",
#                          "512-512-512", "1024-512-256-128"], default="512-256-128")
#   dropout        FloatRange(0.0, 0.4, step=0.05, default=0.05)
#   learning_rate  FloatRange(1e-4, 1e-2, log=True, default=1e-3)
#   weight_decay   FloatRange(1e-6, 1e-2, log=True, default=1e-4)
#   batch_size     Choice([32, 64, 128, 256, 512], default=64)
#
# It is a dict, so narrowing a range is `space["dropout"] = FloatRange(0.0, 0.2)` and dropping
# a knob is `del space["weight_decay"]`. space.to_frame() reads back what will be sampled.
# IntRange / FloatRange / Choice come from workbench.training.hpo_harness.
space = SearchSpace("pytorch")

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    feature_list=features,
    target_column="pec50",
    description="PXR phase-1 pEC50 PyTorch tabular (hyperparameter-searched; phase1_test held out of training)",
    tags=tags,
    hyperparameters={
        "uq_version": "v1",
        # The search budget is what the job costs, so it is worth stating. Everything else
        # defaults: https://supercowpowers.github.io/workbench/models/hpo/
        "hpo": {"n_trials": 100, "search_space": space.to_dict()},
    },
    validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them).
# Features are already in the FeatureSet, so this is a straight scoring pass.
end.inference(phase1[["molecule_name", "smiles", "pec50"] + features], capture_name="pxr_phase1_test")
