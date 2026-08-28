"""The submission model: multi-task Chemprop trained on 100% of the challenge data.

Identical to `cyp_chemprop_mt_all.py` except that nothing is held out. The analog
holdout exists to *choose* between candidates; once chosen, the entry should train on
every labelled row, and the 529 holdout compounds are 11% of the data.

It cannot be scored on the analog holdout — it trained on those rows — so read
`cyp-reg-chemprop-mt` for that number and treat this as the same model with more data.
Its `cv_*` captures are still honest: chemprop's k-fold leaves every row out of the fold
that predicts it, so those are out-of-fold on a scaffold split.

Dropping `validation_ids` is safe — training still runs k-fold and each fold holds out
its own validation for early stopping, so no fold trains without a stopping signal.

Chemprop reads SMILES only, so `openadmet_cyp_f1` and `_f2` are interchangeable here;
f1 keeps it aligned with the model this one is derived from.

Build the FeatureSet first: python cyp_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_f1"
MODEL_NAME = "cyp-reg-chemprop-mt-100"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "submission"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

task_weights = compute_inverse_count_task_weights(df, TARGETS)
print(f"Task weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in task_weights]))}")
print(f"Training on all {len(df)} rows — no holdout")

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=TARGETS,
    description="Multi-task Chemprop over the 4 scored CYP isoforms, trained on 100% for submission",
    tags=TAGS,
    hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()

# Out-of-fold over every row, on chemprop's scaffold split rather than the analog
# holdout. An easier split than the blind set — ST-RAE reads about 1.5x optimistic on
# CYP1A2/CYP2C9, 1.9x on CYP3A4 and 2.5x on CYP2D6 — so read it as a per-isoform ruler
# against the model this one derives from, not as a contest estimate.
end.cross_fold_inference()
