"""PXR phase-1 model: hyperparameter-searched Chemprop on the shared FeatureSet.

Same held-out setup as pxr_chemprop_phase1.py (the phase1_test rows are held out of
training via validation_ids and captured), but the Chemprop knobs are searched rather
than hand-picked. The search runs inside the single training job — trials are ephemeral,
so only the winning config is published as this model.

The search objective is `cv_mae` on scaffold folds of the *training* rows, forced with
hpo["metric"]. Without that override the designated validation rows would become the
objective, which would tune the model on Analog Set 1 and make the pxr_phase1_test
capture optimistic — and unfair against the other phase-1 models, which never see those
labels during fitting.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-hpo-phase1"
tags = ["openadmet_pxr", "chemprop", "hpo", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

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
        "hpo": {
            # "ray" + max_parallel > 1 runs trials concurrently (ASHA) on a 4-GPU instance.
            "backend": "ray",
            "max_parallel": 8,  # 4 GPUs x 2 trials each
            "n_trials": 60,
            # Score on scaffold folds of the training rows, NOT on the held-out phase1_test
            # rows — see the module docstring.
            "metric": "cv_mae",
            # search_space defaults to "basic+lr" (capacity + LR schedule + batch size).
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

# Held-out capture on the phase1_test rows (the model never trained on them)
end.inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
