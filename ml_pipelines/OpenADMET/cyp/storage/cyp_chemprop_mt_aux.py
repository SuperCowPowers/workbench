"""Multi-task Chemprop with single-concentration log2fc as auxiliary targets.

Eight heads on one encoder: the four pIC50 targets we submit, plus the four log2fc
targets that exist only to supervise the shared representation. Same architecture and
same analog holdout as `cyp_chemprop_mt_all.py`, so the A/B is the auxiliary tasks alone.

This targets ranking, which is what is actually left. Per-isoform calibration took macro
ST-RAE from 0.8414 to roughly 0.60 without the model changing, and CYP2C9 and CYP3A4 now
sit at 100% and 98% of the R2 their own Spearman supports -- there is nothing further to
extract there by placement. The gap to the leaders is ordering, concentrated in CYP2D6
(our Spearman 0.432 against the top entry's 0.543), and more encoder supervision is the
one intervention that addresses ordering rather than placement.

The auxiliary targets add 17,500 measurements to the pIC50 track's 6,525, at 89% coverage
against 26-48%, and they span the inactive range the dose-response curves never reach.
OpenADMET's own TabICL baseline routes the same data in through a predicted-log2fc
feature; this is the cheaper form of that idea -- no separate featurizer, no PCA, no
second model.

Task weights hold the baseline's inverse-count values on the four pIC50 heads so the
comparison stays one-variable. The log2fc heads get AUX_WEIGHT relative to the mean pIC50
weight: high enough to shape the encoder, low enough that the submitted heads stay the
objective.

Build the FeatureSet first: python cyp_aux_features.py
"""

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.training.splits import analog_holdout_split
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_aux_f1"
MODEL_NAME = "cyp-reg-chemprop-mt-aux"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "aux_log2fc"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
AUX_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
ALL_TARGETS = TARGETS + AUX_TARGETS
CI_COLUMNS = [f"{t}_{bound}" for t in TARGETS for bound in ("ci_lower", "ci_upper")]

AUX_WEIGHT = 0.3

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

# Derived from the pIC50 targets and the compound set, neither of which the auxiliary
# columns touch — this is the same 529 rows every other CYP model is scored on.
holdout_mask = analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)
holdout = df[holdout_mask]
validation_ids = list(holdout["molecule_name"])

primary_weights = compute_inverse_count_task_weights(df, TARGETS)
aux_weight = AUX_WEIGHT * float(np.mean(primary_weights))
task_weights = list(primary_weights) + [aux_weight] * len(AUX_TARGETS)
print(f"pIC50 weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in primary_weights]))}")
print(f"log2fc weight: {aux_weight:.3f} each ({AUX_WEIGHT} x mean pIC50 weight)")
print(f"Analog holdout: {len(holdout)} of {len(df)} rows held out of training")

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=ALL_TARGETS,
    description="Multi-task Chemprop, 4 pIC50 + 4 single-concentration log2fc auxiliary targets",
    tags=TAGS,
    hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
    validation_ids=validation_ids,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()

# Only the pIC50 targets are scored — the log2fc heads are not a deliverable.
holdout_df = holdout[["molecule_name", "smiles"] + ALL_TARGETS + CI_COLUMNS]
end.inference(holdout_df, capture_name="cyp_analog_holdout")
