"""Multi-task Chemprop over the challenge data plus public ChEMBL and Veith targets.

Eighteen heads on one shared encoder: the four scored pIC50 targets, the four
single-concentration log2fc auxiliaries that already earned their place, five ChEMBL pIC50
targets and five Veith max_response targets. Only the scored four are ever submitted --
the rest exist to shape the representation.

`--public-weight` is the experiment. The proven configuration (`cyp_chemprop_mt_aux_100.py`)
puts auxiliaries at ~23% of total gradient with four heads at `0.3 * mean(primary)`. Ten
more heads at that same per-head weight would put auxiliaries near 51%, outweighing the
targets we are scored on, so whether 0.3 is a per-head number or a per-family budget is an
open question rather than a settled one. Two arms bracket it:

    --public-weight 0.05   ~30% auxiliary share
    --public-weight 0.30   ~51% auxiliary share

`cyp-reg-chemprop-mt-aux-100` is the control at 0% public data. Six-fold apart on the
variable, everything else identical.

Read the arms on OOF against the seed noise floor -- 0.05 Pearson on CYP1A2 and CYP2D6,
0.013 and 0.012 on CYP2C9 and CYP3A4 (`scripts/cyp_seed_noise.py`). Treat a smaller CYP2D6
difference as unresolved. The union's scaffold split runs over 31,670 rows, so the
challenge compounds sit in folds alongside public chemistry where the control's folds were
challenge-only; that is the experiment rather than a flaw, but it is not a seed-for-seed
contrast.

Place its predictions with `scripts/cyp_recalibrate.py --oof MODEL` -- this model has no
board history, so its Pearson comes from its own out-of-fold captures.

Build the FeatureSet first: python cyp_union_features.py
"""

import argparse

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_union_f1"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "public"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
PUBLIC_ISOFORMS = ISOFORMS + ["cyp2c19"]

# Scored targets first: the framework reports a multi-task model on its first target, and
# a bare capture's `prediction` column holds target[0].
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
LOG2FC_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
PUBLIC_TARGETS = ([f"{iso}_pic50_chembl" for iso in PUBLIC_ISOFORMS]
                  + [f"{iso}_max_response" for iso in PUBLIC_ISOFORMS])
ALL_TARGETS = TARGETS + LOG2FC_TARGETS + PUBLIC_TARGETS

# Unchanged from the model this one extends, so the log2fc arm stays a fixed quantity.
LOG2FC_WEIGHT = 0.3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--public-weight", type=float, required=True,
                    help="Per-head weight for the ChEMBL and Veith targets, as a multiple of mean(primary)")
args = parser.parse_args()

model_name = f"cyp-reg-chemprop-union-p{int(round(args.public_weight * 100)):02d}"

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

primary_weights = compute_inverse_count_task_weights(df, TARGETS)
mean_primary = float(np.mean(primary_weights))
log2fc_weight = LOG2FC_WEIGHT * mean_primary
public_weight = args.public_weight * mean_primary
task_weights = (list(primary_weights)
                + [log2fc_weight] * len(LOG2FC_TARGETS)
                + [public_weight] * len(PUBLIC_TARGETS))

aux_share = (log2fc_weight * len(LOG2FC_TARGETS) + public_weight * len(PUBLIC_TARGETS)) / sum(task_weights)
print(f"Building {model_name} on all {len(df):,} rows — no holdout")
print(f"pIC50 weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in primary_weights]))}")
print(f"log2fc weight: {log2fc_weight:.3f} each ({LOG2FC_WEIGHT} x mean primary)")
print(f"public weight: {public_weight:.3f} each ({args.public_weight} x mean primary), {len(PUBLIC_TARGETS)} heads")
print(f"auxiliary share of total gradient: {100 * aux_share:.0f}%")

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=ALL_TARGETS,
    description=f"Multi-task Chemprop, challenge + public targets, public weight {args.public_weight}",
    tags=TAGS + [f"public_weight_{args.public_weight}"],
    hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()

# Out-of-fold over every row. The scored targets exist on 4,905 of them, which is the
# subset the arms are compared on.
end.cross_fold_inference()
