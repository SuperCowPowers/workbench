"""PXR phase-1 model: hyperparameter-searched Chemprop on the shared FeatureSet.

Same held-out setup as pxr_chemprop_phase1.py (the phase1_test rows are held out of
training via validation_ids and captured), but the Chemprop knobs are searched rather
than hand-picked. The search runs inside the single training job — trials are ephemeral,
so only the winning config is published as this model.

The search objective is `cv_mae` on scaffold folds of the *training* rows. The phase1_test
rows are held out of training and never scored during the search, so the pxr_phase1_test
capture stays an honest comparison against the other phase-1 models.

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
#   ffn_hidden_dim  Choice(["300", "600", "1200", "1800", "300-300", "300-100", ...])
#   max_lr          FloatRange(1e-4, 5e-3, log=True, default=1e-3)
#   batch_size      Choice([64, 128, 256, 512], default=64)
#
# ffn_hidden_dim is a per-layer shape: its length is the head's depth, so it covers what
# chemprop splits across ffn_hidden_dim + ffn_num_layers, plus the tapered heads.
#
# It is a dict, so narrowing a range is `space["depth"] = IntRange(3, 5)` and dropping a knob
# is `del space["depth"]`. space.to_frame() reads back what will be sampled.
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
        # The search budget is what the job costs, so it is worth stating. Everything else
        # defaults: https://supercowpowers.github.io/workbench/models/hpo/
        "hpo": {"n_trials": 60, "search_space": space.to_dict()},
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
