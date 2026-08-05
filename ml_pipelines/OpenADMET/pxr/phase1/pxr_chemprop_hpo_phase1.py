"""PXR phase-1 model: hyperparameter-searched Chemprop on the shared FeatureSet.

Same held-out setup as pxr_chemprop_phase1.py (the phase1_test rows are held out of
training via validation_ids and captured), but the Chemprop knobs are searched rather
than hand-picked. The search runs inside the single training job — trials are ephemeral,
so only the winning config is published as this model.

The search objective is `cv_mae` on scaffold folds of the *training* rows — the default,
and the one that matters here. Setting hpo["metric"]="holdout_mae" would tune the model on
Analog Set 1 and make the pxr_phase1_test capture optimistic, and unfair against the other
phase-1 models, which never see those labels during fitting.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.training.hpo_harness import SearchSpace

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-hpo-phase1"
tags = ["openadmet_pxr", "chemprop", "hpo", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

# The knobs to search — Chemprop's shipped space, unchanged:
#
#   depth           IntRange(2, 6, step=1, default=5)
#   hidden_dim      IntRange(100, 2400, step=100, default=700)
#   ffn_num_layers  IntRange(1, 3, step=1, default=2)
#   ffn_hidden_dim  Choice([300, 600, 1800, "300-100", "512-128", "512-128-32", "1024-256-64"])
#   max_lr          FloatRange(1e-4, 5e-3, log=True, default=1e-3)
#   batch_size      Choice([64, 128, 256, 512], default=64)
#
# It is a dict, so narrowing a range is `space["depth"] = IntRange(3, 5)` and dropping a knob
# is `del space["ffn_num_layers"]`. space.to_frame() reads back what will be sampled.
# IntRange / FloatRange / Choice come from workbench.training.hpo_harness.
space = SearchSpace("chemprop")

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column="pec50",
    description="PXR phase-1 pEC50 Chemprop (hyperparameter-searched; phase1_test held out of training)",
    tags=tags,
    hyperparameters={
        "uq_version": "v1",
        # The search budget is what the job costs, so it is worth stating. Everything else
        # defaults: https://supercowpowers.github.io/workbench/models/hpo/
        "hpo": {"n_trials": 40, "search_space": space.to_dict()},
    },
    validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
end.inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
