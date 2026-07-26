"""PXR phase-1 multi-task Chemprop with a hyperparameter search — pEC50 + logD.

The HPO counterpart to the `mt-logd` variant in pxr_chemprop_mt_phase1.py, and the
smallest multi-task combination available: logD carries ~4k auxiliary rows against
logP's ~52k, so this searches in a fraction of the time a logP variant would.

pEC50 is the primary task and the only one scored; logD supervises the shared MPNN
encoder. Two things make the multi-task search honest, both of which the single-task
HPO path did not need:

  - The objective is the PRIMARY target's MAE via `nanmean`. The training view keeps a
    row when any target is non-NaN, so logD-only rows have no pEC50 — they train the
    encoder but are skipped when scoring.
  - `task_weights` is forwarded into the search. Without it every trial would train at
    an equal task weight and the winner would be selected against a loss the published
    model never uses.

The search objective is `cv_mae` on scaffold folds of the training rows (the default).
The phase1_test rows stay out of training AND out of the search, so the capture below
is directly comparable to `pxr-reg-chemprop-mt-logd` and the single-task models.

Build the FeatureSet first: python ../pxr_chemprop_mt_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType

fs_name = "openadmet_pxr_mt"
model_name = "pxr-reg-chemprop-mt-logd-hpo"
tags = ["openadmet_pxr", "chemprop", "multi_task", "mt-logd", "hpo", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=["pec50", "logd"],  # pec50 first (primary); logd is the aux head
    description="PXR phase-1 multi-task Chemprop, aux=logd, hyperparameter-searched (phase1_test held out)",
    tags=tags,
    hyperparameters={
        "uq_version": "v1",
        "task_weights": [1.0, 0.3],  # matches pxr-reg-chemprop-mt-logd, so the comparison is clean
        "hpo": {
            "backend": "ray",
            # One trial per GPU. The multi-task view carries roughly twice the rows of the
            # single-task one, and the good configs here sit at the top of the capacity
            # range — packing two per card runs them out of memory.
            "max_parallel": 4,
            "gpus_per_trial": 1.0,
            "n_trials": 60,
            # Score on scaffold folds of the training rows, never on the held-out
            # phase1_test rows — tuning on those would cost them their role as a benchmark.
            "metric": "cv_mae",
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
# `prediction` aliases the primary (pec50) head, so it's comparable to the baseline.
end.inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
