"""PXR phase-1 model: hyperparameter-searched PyTorch tabular on the shared FeatureSet.

Same held-out setup as the other phase-1 models (the phase1_test rows are held out of
training via validation_ids and captured), on the 2D descriptor columns. The search runs
inside the single training job — trials are ephemeral, so only the winning config is
published as this model.

The search objective is `cv_mae` on scaffold folds of the *training* rows, forced with
hpo["metric"]. Without that override the designated validation rows would become the
objective, which would tune the model on Analog Set 1 and make the pxr_phase1_test
capture optimistic — and unfair against the other phase-1 models, which never see those
labels during fitting.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

import logging

from workbench.api import Endpoint, FeatureSet, ModelFramework, ModelType

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
        "hpo": {
            # "ray" + max_parallel > 1 runs trials concurrently (ASHA) on a 4-GPU instance.
            "backend": "ray",
            "max_parallel": 8,  # 4 GPUs x 2 trials each
            "n_trials": 100,
            # Score on scaffold folds of the training rows, NOT on the held-out phase1_test
            # rows — see the module docstring.
            "metric": "cv_mae",
            # search_space defaults to "basic+lr" (layer shape + dropout, optimizer, batch size).
            # The search only shortlists: its top rerank_top_k configs are re-scored against
            # these hyperparameters as-is, and the winner of *that* is published. So a search
            # that finds nothing real publishes the untuned baseline.
            "rerank_top_k": 5,
        },
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
