"""Multi-task Chemprop over the four challenge CYP isoforms plus CYP2C19.

A dry run of the challenge model shape, on public Veith data, so the plumbing is
proven before the real training set lands. CYP3A4 is the primary task; the other
isoforms supervise the shared MPNN encoder. Coverage is uneven (CYP1A2 has ~9.6k
curves, CYP2D6 ~5.7k, and only ~2k compounds carry all five), so `task_weights`
come from inverse task counts rather than being hand-set.

Every isoform here is an end product, not an auxiliary — the weighting corrects
coverage imbalance, it does not demote a task.

Evaluation is the analog holdout, not cross-validation: `analog_holdout_split`
reproduces how the challenge test set was built (top hits plus their nearest
neighbors), and on this data a random split of the same size flatters a baseline by
about 2x. Those rows are held out of training via `validation_ids` and captured
separately.

Build the FeatureSet first: python cyp_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.training.splits import analog_holdout_split
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_veith"
MODEL_NAME = "cyp-reg-chemprop-mt"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity"]

# CYP3A4 primary (it metabolises roughly half of marketed drugs); CYP2C19 last since
# the challenge does not score it.
TARGETS = ["cyp3a4_pic50", "cyp2c9_pic50", "cyp2d6_pic50", "cyp1a2_pic50", "cyp2c19_pic50"]

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

# Hold out potent hits and their close analogs across every target, mirroring the
# challenge's hit-expansion test set.
holdout_mask = analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)
holdout = df[holdout_mask]
validation_ids = list(holdout["id"])

task_weights = compute_inverse_count_task_weights(df[TARGETS].to_numpy())
print(f"Task weights: {dict(zip(TARGETS, [round(float(w), 3) for w in task_weights]))}")
print(f"Analog holdout: {len(holdout)} of {len(df)} rows held out of training")

model = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=TARGETS,
    description="Multi-task Chemprop over 5 CYP isoforms (public Veith panel), analog holdout",
    tags=TAGS,
    hyperparameters={"task_weights": list(task_weights), "uq_version": "v1"},
    validation_ids=validation_ids,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()
# Held-out capture on exactly the analog rows the model never trained on.
end.inference(holdout[["id", "smiles"] + TARGETS], capture_name="cyp_analog_holdout")
