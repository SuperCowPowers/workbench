"""Multi-task Chemprop over the four scored CYP isoforms — the challenge model shape.

One shared MPNN encoder over all 6,525 dose-response measurements rather than four
models on ~1,500 rows each. That pooling is chemprop's actual claim on this challenge:
the isoforms are correlated and the per-isoform data is small. It is not that graph
learning resolves activity cliffs — descriptor models match or beat GNNs there — so
the comparison worth running is this against single-task chemprop, not against XGBoost.

Every isoform here is an end product, not an auxiliary. Coverage is uneven (CYP3A4 has
2,335 curves, CYP2C9 only 1,285), so `task_weights` come from inverse task counts to
correct the imbalance, which does not demote a task.

Evaluation is the analog holdout, not cross-validation: `analog_holdout_split`
reproduces how the challenge test set was built (top hits plus their nearest
neighbors), and a random split of the same size flatters a baseline by about 2x.
Those rows are held out of training via `validation_ids` and captured separately.

`feature_list=["smiles"]` is explicit and load-bearing. The FeatureSet carries each
target's `_ci_lower`/`_ci_upper` so ST-RAE can be scored against them; an
auto-generated feature list would hand the model the bounds bracketing its own label.

The primary-target counterpart is `cyp_chemprop_mt.py` (one model per isoform, that
isoform weighted 1.0 and the rest 0.3). Both scripts use `openadmet_cyp_f1` and the same
50/10 analog holdout, so their per-target captures are directly comparable.

Build the FeatureSet first: python cyp_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.training.splits import analog_holdout_split
from workbench.utils.multi_task import compute_inverse_count_task_weights

from pipeline_utils.cyp_scoring import capture_st_rae

FS_NAME = "openadmet_cyp_f1"
MODEL_NAME = "cyp-reg-chemprop-mt"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity"]

# CYP3A4 first (it metabolises roughly half of marketed drugs) — all four are scored.
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
CI_COLUMNS = [f"{t}_{bound}" for t in TARGETS for bound in ("ci_lower", "ci_upper")]

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

# Hold out potent hits and their close analogs across every target, mirroring the
# challenge's hit-expansion test set.
holdout_mask = analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)
holdout = df[holdout_mask]
validation_ids = list(holdout["molecule_name"])

task_weights = compute_inverse_count_task_weights(df, TARGETS)
print(f"Task weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in task_weights]))}")
print(f"Analog holdout: {len(holdout)} of {len(df)} rows held out of training")

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=TARGETS,
    description="Multi-task Chemprop over the 4 scored CYP isoforms, analog holdout",
    tags=TAGS,
    hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
    validation_ids=validation_ids,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on exactly the analog rows the model never trained on. The CI columns
# ride along so ST-RAE is scored against the challenge's own credible intervals.
holdout_df = holdout[["molecule_name", "smiles"] + TARGETS + CI_COLUMNS]
end.inference(holdout_df, capture_name="cyp_analog_holdout")

# ST-RAE is the challenge's metric but not a Workbench one, so it is scored here from the
# capture's own predictions against the credible intervals carried in the FeatureSet.
print(f"Analog-holdout ST-RAE:\n{capture_st_rae(model, TARGETS).to_string(index=False)}")
