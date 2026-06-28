"""PXR phase-1 multi-task Chemprop — does logP / logD as auxiliary tasks help?

Three multi-task variants off the shared multi-task FeatureSet (openadmet_pxr_mt):
pEC50 is always the primary task; the auxiliaries supervise the shared MPNN encoder
but only the pEC50 head is scored. Per-variant `target_column` subsetting excludes
the unused auxiliary (its rows become all-NaN and chemprop masks them out):

  - mt-logp : pEC50 + logP   (data-rich, 52k aux rows — the OOD-transfer bet)
  - mt-logd : pEC50 + logD   (mechanistic, pH-7.4 ionization)
  - mt-both : pEC50 + logP + logD

`task_weights` keep pEC50 dominant in the gradient (logP is down-weighted harder
since it has ~13x the rows). Each variant zero-weights the phase1_test rows out of
training and captures `pxr_phase1_test` on exactly those rows, so the held-out RAE
is directly comparable to the single-task baseline (~0.569).

Build the FeatureSet first: python ../pxr_chemprop_mt_feature_sets.py
"""

from workbench.api import FeatureSet, Model, Endpoint, ModelType, ModelFramework

recreate = False
fs_name = "openadmet_pxr_mt"
base_tags = ["openadmet_pxr", "chemprop", "multi_task", "phase1"]

# (suffix, target_column [primary first], task_weights) — pec50 always primary.
VARIANTS = [
    ("logp", ["pec50", "logp"], [1.0, 0.2]),
    ("logd", ["pec50", "logd"], [1.0, 0.3]),
    ("both", ["pec50", "logp", "logd"], [1.0, 0.2, 0.3]),
]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]
weights = {mid: 0.0 for mid in phase1["molecule_name"]}  # held-out rows don't train

for suffix, targets, task_weights in VARIANTS:
    model_name = f"pxr-reg-chemprop-mt-{suffix}"
    tags = base_tags + [f"mt-{suffix}"]
    if recreate or not Model(model_name).exists():
        m = fs.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            feature_list=["smiles"],
            target_column=targets,  # pec50 first (primary); aux head(s) follow
            description=f"PXR phase-1 multi-task Chemprop, aux={'+'.join(targets[1:])} (phase1_test zero-weighted)",
            tags=tags,
            hyperparameters={"uq_version": "v1", "task_weights": task_weights},
            sample_weights=weights,
        )
        m.set_owner("open_admet_pxr")
        end = m.to_endpoint(tags=tags)
        end.set_owner("open_admet_pxr")
        end.test_inference()
        end.cross_fold_inference()
    # Held-out capture on the phase1_test rows (the model never trained on them).
    # `prediction` aliases the primary (pec50) head, so it's comparable to the baseline.
    Endpoint(model_name).inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
