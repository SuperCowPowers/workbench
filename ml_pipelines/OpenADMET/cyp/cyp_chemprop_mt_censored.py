"""Multi-task Chemprop with censored CYP2D6 labels — the A/B against `cyp-reg-chemprop-mt`.

Same architecture, same targets, same analog holdout as `cyp_chemprop_mt_all.py`. The
only differences are the FeatureSet (`openadmet_cyp_censored_f1`, which carries 2,627
left-censored CYP2D6 rows the fitted-curve set drops) and `bounded_loss=True`, which is
what makes chemprop read them.

This did not work, and the mechanism is worth keeping. Bounded loss has zero gradient
below the bound, so the cheapest way to satisfy 2,627 rows all bounded at the same value
is to predict a constant just under it. That is what the model did: on those rows it
moved from sd 0.19 to **sd 0.07**, mean 4.49, with 96.5% under the bound. The constraint
was honoured perfectly and taught nothing.

Because those rows are "every compound that is not a CYP2D6 inhibitor" -- a large and
chemically diverse region -- that constant propagated through the shared head. Blind-set
CYP2D6 predictions went from sd 0.41 to 0.30 and the floor rose from 3.40 to 4.27, with
the mean barely moving (4.69 -> 4.62). Narrower and no better centred, which lowers R2.
On the analog holdout macro ST-RAE moved -0.001, inside the +-0.03 noise floor.

The root cause is in the assay, not the loss. Median log2fc by fitted-pIC50 bin is flat
for CYP2D6 across the low end (-1.07 / -0.98 / -0.94 / -1.11 for bins 2-3.5 through
4.31-4.6) where CYP3A4's is monotone (-0.18 / -0.77 / -1.32 / -2.03). The
single-concentration arm can say "not a CYP2D6 inhibitor" but carries no information
about how far below the bound a compound sits, so neither censoring nor imputing from it
can add low-end range. The one isoform that needed this is the one isoform the readout
cannot serve.

Do not extend this to CYP1A2 on the strength of its row counts alone -- check the
flatness of its log2fc calibration first.

Task weights come from the fitted-label counts, not the censored ones. Inverse-count
weighting would otherwise read CYP2D6 as the best-covered task and demote it, which is
backwards, and holding the weights at the baseline's values keeps this a one-variable
comparison.

Read `cyp-reg-chemprop-mt` for the baseline (macro ST-RAE 0.702, CYP2D6 Pearson 0.419).

Build the FeatureSet first: python cyp_censored_features.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.training.splits import analog_holdout_split
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_censored_f1"
MODEL_NAME = "cyp-reg-chemprop-mt-cen"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "censored"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
CI_COLUMNS = [f"{t}_{bound}" for t in TARGETS for bound in ("ci_lower", "ci_upper")]
CENSORED_TARGET = "cyp2d6_pic50_direct_inhibition"

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

holdout_mask = analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)
holdout = df[holdout_mask]
validation_ids = list(holdout["molecule_name"])

# Weights from fitted labels only — see the module docstring.
fitted_only = df[~df[f"{CENSORED_TARGET}_lt"].fillna(False).astype(bool)]
task_weights = compute_inverse_count_task_weights(fitted_only, TARGETS)
print(f"Task weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in task_weights]))}")
print(f"Censored CYP2D6 rows: {int(df[f'{CENSORED_TARGET}_lt'].sum()):,}")
print(f"Analog holdout: {len(holdout)} of {len(df)} rows held out of training")

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=TARGETS,
    description="Multi-task Chemprop, CYP2D6 left-censored at 4.6, analog holdout",
    tags=TAGS,
    hyperparameters={"task_weights": task_weights, "uq_version": "v1", "bounded_loss": True},
    validation_ids=validation_ids,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()

# The holdout carries only fitted labels, so this capture is directly comparable to the
# baseline's despite the training set having changed.
holdout_df = holdout[["molecule_name", "smiles"] + TARGETS + CI_COLUMNS]
end.inference(holdout_df, capture_name="cyp_analog_holdout")

metrics = model.get_inference_metrics(capture_name="cyp_analog_holdout")
if metrics is not None and "st_rae" in metrics.columns:
    print(f"Analog-holdout ST-RAE: {metrics[['st_rae']].to_string(index=False)}")
else:
    cols = None if metrics is None else metrics.columns.tolist()
    print(f"st_rae MISSING from inference metrics (columns: {cols}) — credible intervals did not survive")
