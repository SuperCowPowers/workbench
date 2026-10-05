"""The single-concentration screen as its own model, to be read out as a feature.

The screen is the CYP analogue of the primary screen PXR drew its readouts from: same lab,
same compounds, 4,375 labels per isoform against 1,285-2,335 scored ones. Measured log2FC
orders the scored CYP2D6 labels at rho -0.83, so a model that predicts it well carries
something the scored column does not.

A readout is that prediction carried as a *feature*, which is the point. Auxiliary heads have
failed here three times for one reason -- a head keeps its own scale, so low-range information
never crosses encoder -> head -> head -- and a feature column bypasses that path entirely.

**Nothing here sees a pIC50.** `cyp-reg-chemprop-mt-aux-100` also carries log2FC heads, but
its predictions correlate -0.75 with its own pIC50 output and only +0.42 with the measured
log2FC, so it distils itself rather than the screen. Four heads and one encoder, screen only,
is what makes the readout an independent view.

The screen is disjoint from the blind set (0 of 748 skeletons), so every blind compound's
readout comes from the endpoint rather than from a row this model trained on.

Consumed by `scripts/cyp_readout_spike.py` and the readout FeatureSet.

Build the FeatureSet first:  ml_pipeline_launcher cyp_aux_features
"""

import argparse

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_aux_f1"
MODEL_NAME = "cyp-reg-chemprop-log2fc"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "screen", "readout"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--name-suffix",
    default=None,
    help="Append to the model name, so a rebuild lands beside the existing model rather than replacing it. "
    "A retrain redraws the fold split, so its out-of-fold numbers move within seed noise",
)
args = parser.parse_args()
model_name = MODEL_NAME + (f"-{args.name_suffix.strip('-')}" if args.name_suffix else "")

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

# All four arms are end products here, so unequal coverage is corrected symmetrically.
task_weights = compute_inverse_count_task_weights(df, TARGETS)
labelled = {t.split("_")[0]: int(df[t].notna().sum()) for t in TARGETS}
print(f"log2fc labels per isoform: {labelled}")
print(f"task weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in task_weights]))}")

# Rows the screen never measured carry no target at all and leave the training view.
unmeasured = list(df.loc[df[TARGETS].notna().sum(axis=1).eq(0), "molecule_name"])
print(f"Training on {len(df) - len(unmeasured):,} screened rows; excluding {len(unmeasured):,} unscreened")

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=TARGETS,
    description="Single-concentration log2FC, four isoforms — the readout model",
    tags=TAGS,
    hyperparameters={"task_weights": [float(w) for w in np.asarray(task_weights)]},
    exclude_ids=unmeasured,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()
