"""The submission model: auxiliary-target Chemprop trained on 100% of the challenge data.

Identical to `cyp_chemprop_mt_aux.py` except that nothing is held out. The analog holdout
exists to *choose* between candidates; once chosen, the entry should train on every
labelled row, and the 529 holdout compounds are 11% of the data.

`cyp_chemprop_mt_aux.py` is what earned this a submission: on the analog holdout it lifted
CYP2D6 Pearson 0.419 -> 0.502 and CYP1A2 0.588 -> 0.619 against `cyp-reg-chemprop-mt`,
with every isoform improving. That is a ranking gain, which is the only thing left --
per-isoform calibration has taken CYP2C9 and CYP3A4 to 100% and 98% of the R2 their own
Spearman supports.

It cannot be scored on the analog holdout, having trained on those rows. Read
`cyp-reg-chemprop-mt-aux` for that. Its `cv_*` captures are still honest: chemprop's
k-fold leaves every row out of the fold that predicts it, so those are out-of-fold on a
scaffold split, reading roughly 1.5x optimistic on CYP1A2/CYP2C9, 1.9x on CYP3A4 and 2.5x
on CYP2D6 against the board.

Predictions from this model need their own calibration. The offsets fitted for
`cyp-reg-chemprop-mt-100` do not transfer -- different spread means different bias.
Use `scripts/cyp_recalibrate.py --moments` against the estimated blind-set distribution
rather than spending a submission to rediscover it.

Build the FeatureSet first: python cyp_aux_features.py
"""

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_aux_f1"
MODEL_NAME = "cyp-reg-chemprop-mt-aux-100"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "aux_log2fc", "submission"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
AUX_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
ALL_TARGETS = TARGETS + AUX_TARGETS

AUX_WEIGHT = 0.3

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

primary_weights = compute_inverse_count_task_weights(df, TARGETS)
aux_weight = AUX_WEIGHT * float(np.mean(primary_weights))
task_weights = list(primary_weights) + [aux_weight] * len(AUX_TARGETS)
print(f"pIC50 weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in primary_weights]))}")
print(f"log2fc weight: {aux_weight:.3f} each ({AUX_WEIGHT} x mean pIC50 weight)")
print(f"Training on all {len(df)} rows — no holdout")

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=ALL_TARGETS,
    description="Multi-task Chemprop with log2fc auxiliary targets, trained on 100% for submission",
    tags=TAGS,
    hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()

# Out-of-fold over every row, on chemprop's scaffold split rather than the analog holdout.
end.cross_fold_inference()
