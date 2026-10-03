"""CYP2D6 TabICL on the Monroe embedding -- does a different representation move ranking?

CYP2D6 is where the entry is weakest and where a win is large enough to measure. On the
interim board (all 750 compounds) our Spearman is 0.468 against a leader at 0.598; out of
fold the resolvable difference is 0.056, so a gap that size is readable where our usual
0.01-0.03 model deltas are not.

Ten attempts have failed to move it -- task weighting, a dedicated encoder, single-task,
fingerprints, binary heads, Tox21 potency heads, a log2fc surrogate, empirical-Bayes
shrinkage, an uncertainty tilt, low-band loss weighting, pooled public labels, and the
CheMeleon pretrained encoder. Every one of those changed the target, the loss, or the data.
This changes the representation, which is the remaining class: TabICL over Monroe rather
than a D-MPNN over the molecular graph.

The band to watch is separation, not within-band ordering. Our AUC for telling a sub-4.0
compound from a >=4.5 one is 0.750 against 0.939 on CYP3A4 -- a 3.5-log distinction with no
label-noise excuse, and the part of CYP2D6 that is actually wrong.

Nothing drops NaN-target rows on the way into a training view, and CYP2D6 carries 1,493
labels in a 4,905-row set, so the unlabelled rows are handed to `exclude_ids`. Any
single-target model over these FeatureSets needs the same treatment -- Chemprop gets away
without it because its multi-task loss masks missing targets.

`top_variance_features` keeps the 100 highest-variance embedding columns. On PXR 100 / 360 /
720 scored within the noise of 253 compounds, and 100 needs the least serving memory.

TabICL is single-target, so this is CYP2D6 alone. Multi-task is cheap to lose here:
`mt-aux-100` against `union-p30-h26` is -0.002 mean across four isoforms, and `2d6-single`
against `2d6-isoform` sits inside its own threshold.

Read it with `scripts/cyp_compare.py cyp-reg-tabicl-2d6-monroe cyp-reg-chemprop-2d6-single --bands`
against the 0.056 threshold. Out-of-fold baselines on the same 1,493 rows: 0.388 single,
0.445 isoform, 0.503 for the four-model ensemble.

Build the FeatureSet first:  ml_pipeline_launcher cyp_monroe_feature_sets
"""

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_monroe_f1"
MODEL_NAME = "cyp-reg-tabicl-2d6-monroe"
TARGET = "cyp2d6_pic50_direct_inhibition"
TAGS = ["openadmet_cyp", "tabicl", "monroe", "activity"]

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()[["molecule_name", TARGET]]
unlabelled = list(df.loc[df[TARGET].isna(), "molecule_name"])
print(f"Training on {len(df) - len(unlabelled):,} labelled rows; excluding {len(unlabelled):,} without a CYP2D6 label")

m = fs.to_model(
    name=MODEL_NAME,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.TABICL,
    feature_list=["monroe"],
    target_column=TARGET,
    description="CYP2D6 pIC50, TabICL over the Monroe embedding",
    tags=TAGS,
    hyperparameters={"top_variance_features": 100},
    exclude_ids=unlabelled,
)
m.set_owner("open_admet_cyp")
print(f"Measured serving memory: {(m.workbench_meta() or {}).get('workbench_inference_memory_gb')} GB")

end = m.to_endpoint(tags=TAGS)
end.set_owner("open_admet_cyp")
end.test_inference()
end.cross_fold_inference()
