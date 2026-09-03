"""Multi-task Chemprop over the challenge data plus public ChEMBL and Veith targets.

Eighteen heads on one shared encoder: the four scored pIC50 targets, the four
single-concentration log2fc auxiliaries that already earned their place, five ChEMBL pIC50
targets and five Veith max_response targets. Only the scored four are ever submitted --
the rest exist to shape the representation.

`--public-weight` is the experiment, and it moves more than its name suggests.

**A task's gradient share is its label count times its weight, not its share of the task
list.** chemprop multiplies the per-element loss by `task_weights` and reduces by summing
over observed labels (`nn/metrics.py`: `L * weights * task_weights * mask`, then
`total_loss / mask.sum()`), so a task with twenty times the labels dominates a task with the
same weight. The public targets carry 137,375 labels against the four scored targets' 6,525,
which is why a 0.3 multiplier on them is not a modest thing:

    --public-weight 0.05    scored 27%   assay 42%   public 30%
    --public-weight 0.30    scored 11%   assay 17%   public 72%

For reference the proven configuration (`cyp_chemprop_mt_aux_100.py`, four scored targets
plus four log2fc) runs the scored targets at 54%. At `--public-weight 0.30` this script puts
them at 11%, so the model is mostly fitting ChEMBL and Veith and the outputs we are scored on
are a minority interest. Whether that is the point or the problem is the experiment.

`--tox21` shifts these by about a point (26.9% / 10.4% scored), since it adds tasks that are
small next to the ChEMBL and Veith label counts already present.

These shares depend on the label counts in the FeatureSet, not just on the flag, so they move
whenever a source is added or re-pulled. Read the value the script prints at build time.

`--tox21` adds Tox21's CYP potency as five more heads at the same public weight. What it
brings is range rather than volume: its actives are weak ones, median pIC50 4.76, where
ChEMBL bottoms out at 4.0 and the challenge set is hit-enriched. Out-of-fold the model
predicts above the credible ceiling for 126 of CYP2D6's 129 sub-4.0 rows, so low-end
examples are the thing it has never had. Contrast against the same `--public-weight`
without the flag -- one variable, though the extra rows do move the scaffold folds.

`--censored` trains on `openadmet_cyp_union_censored_f1`, where ChEMBL's `IC50 > x` records
arrive as bounds, and turns on `bounded_loss`. A bound then costs the model only when its
prediction rises above it, which is what lets the ChEMBL heads stop punishing a low
prediction -- the uncensored file has no label under pIC50 4.0 at all, while 9% of the
challenge's CYP2D6 labels and 40% of its CYP3A4 ones sit there.

Bounds are 15-29% of each ChEMBL head's labels -- 29% on CYP2D6, so that head still has 71%
real measurements to fix its scale -- and they are spread, because each record carries its own
reported cutoff: 74-109 distinct values per isoform, with 29-35% of bounds on the most
common one. That spread is the point. Bounded loss has no gradient below the bound, so a head
whose rows are mostly one repeated bound is cheapest satisfied by a constant just under it.
The qHTS panels are exactly that shape -- 53-77% of their bounds sit on a single value, and
Veith's CYP2D6 arm is 11,118 bounds over three -- and are deliberately not in this variant.

Three variants read together, one variable at a time:

    --public-weight W                 uncensored ChEMBL, bounds absent
    --public-weight W --censored      bounds honoured by the loss
    --public-weight W --censored --bounds-as-labels    bounds read as exact measurements

The third is the control that shows the loss is doing the work rather than the extra rows.

Compare only models built from the same target set. The name is derived from the flags, so
it does not record how many heads were in the script on the day a model was built, and this
script has gained heads over time -- `cyp-reg-chemprop-union-p30` carries 18 against the
26 a bare `--public-weight` run produces now. `--name-suffix` keeps a rebuild from replacing
an older model that meant something else, and the run warns when the head counts differ.

`cyp-reg-chemprop-mt-aux-100` is the control at 0% public data. Six-fold apart on the
variable, everything else identical.

Read the variants on OOF against the seed noise floor -- 0.05 Pearson on CYP1A2 and CYP2D6,
0.013 and 0.012 on CYP2C9 and CYP3A4 (`scripts/cyp_seed_noise.py`). Treat a smaller CYP2D6
difference as unresolved. The union's scaffold split runs over 31,670 rows, so the
challenge compounds sit in folds alongside public chemistry where the control's folds were
challenge-only; that is the experiment rather than a flaw, but it is not a seed-for-seed
contrast.

Place its predictions with `scripts/cyp_recalibrate.py --oof MODEL` -- this model has no
board history, so its Pearson comes from its own out-of-fold captures.

Build the FeatureSet first: python cyp_union_features.py [--censored]
"""

import argparse

import numpy as np
from workbench.api import FeatureSet, Model, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

UNCENSORED_FS = "openadmet_cyp_union_f1"
CENSORED_FS = "openadmet_cyp_union_censored_f1"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "public"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
PUBLIC_ISOFORMS = ISOFORMS + ["cyp2c19"]

# Scored targets first: the framework reports a multi-task model on its first target, and
# a bare capture's `prediction` column holds target[0].
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
LOG2FC_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
# The assay's other arms, same platform and chemistry as the scored targets.
TDI_TARGETS = [f"{iso}_pic50_tdi_condition" for iso in ISOFORMS]
EMAX_TARGETS = [f"{iso}_emax_vs_pos_ctrl_direct_inhibition" for iso in ISOFORMS]
ASSAY_TARGETS = LOG2FC_TARGETS + TDI_TARGETS + EMAX_TARGETS
PUBLIC_TARGETS = [f"{iso}_pic50_chembl" for iso in PUBLIC_ISOFORMS] + [f"{iso}_max_response" for iso in PUBLIC_ISOFORMS]
# Tox21's CYP potency on its own scale. The reason it is here is range, not volume: its
# actives are weak ones, median pIC50 4.76, where ChEMBL stops at 4.0 and the challenge set
# is hit-enriched. It also reaches ~5.6k compounds neither other source covers, and covers
# them where the blind set is furthest from our training chemistry.
TOX21_TARGETS = [f"{iso}_pic50_tox21" for iso in PUBLIC_ISOFORMS]

# One weight for every challenge-assay auxiliary, at the value the log2fc heads were validated
# at, so adding heads does not silently re-tune the configuration that already worked.
ASSAY_WEIGHT = 0.3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--public-weight",
    type=float,
    required=True,
    help="Per-head weight for the ChEMBL and Veith targets, as a multiple of mean(primary)",
)
parser.add_argument(
    "--tox21",
    action="store_true",
    help="Add the Tox21 potency heads at the same public weight",
)
parser.add_argument(
    "--censored",
    action="store_true",
    help="Train on the censored FeatureSet with bounded_loss, so ChEMBL's IC50>x records act as bounds",
)
parser.add_argument(
    "--bounds-as-labels",
    action="store_true",
    help="Control variant: take the censored FeatureSet but leave bounded_loss off, so bounds read as exact labels",
)
parser.add_argument(
    "--name-suffix",
    default=None,
    help="Append to the model name. The name is derived from the flags, not the head count, "
    "so rebuilding after heads are added silently replaces a model that meant something else",
)
args = parser.parse_args()
if args.bounds_as_labels and not args.censored:
    parser.error("--bounds-as-labels only means something with --censored")

public_targets = PUBLIC_TARGETS + (TOX21_TARGETS if args.tox21 else [])
all_targets = TARGETS + ASSAY_TARGETS + public_targets

model_name = f"cyp-reg-chemprop-union-p{int(round(args.public_weight * 100)):02d}"
if args.tox21:
    model_name += "-tox"
if args.censored:
    model_name += "-cenlabels" if args.bounds_as_labels else "-cen"
if args.name_suffix:
    model_name += f"-{args.name_suffix.strip('-')}"

bounded_loss = args.censored and not args.bounds_as_labels

# A name derived from flags alone cannot distinguish head counts, so say plainly when this
# run would replace a model that trained on a different target set.
existing = Model(model_name)
if existing.exists():
    prior = existing.target()
    prior = list(prior) if isinstance(prior, list) else [prior]
    if len(prior) != len(all_targets):
        print(
            f"WARNING: '{model_name}' exists with {len(prior)} targets; this run has {len(all_targets)}. "
            f"Replacing it makes any comparison against the old one confounded — use --name-suffix instead."
        )

fs = FeatureSet(CENSORED_FS if args.censored else UNCENSORED_FS)
df = fs.pull_dataframe()

primary_weights = compute_inverse_count_task_weights(df, TARGETS)
mean_primary = float(np.mean(primary_weights))
assay_weight = ASSAY_WEIGHT * mean_primary
public_weight = args.public_weight * mean_primary
task_weights = list(primary_weights) + [assay_weight] * len(ASSAY_TARGETS) + [public_weight] * len(public_targets)


def gradient_share(columns: list[str], weights: list[float]) -> tuple[int, float]:
    """Labels, and the loss mass they carry.

    chemprop weights the per-element loss and sums over observed labels, so a task
    contributes in proportion to its label count times its weight. Counting tasks instead
    understates a group holding twenty times the labels at the same weight.
    """
    labels = sum(int(df[c].notna().sum()) for c in columns if c in df.columns)
    mass = sum(int(df[c].notna().sum()) * w for c, w in zip(columns, weights) if c in df.columns)
    return labels, mass


groups = {
    "scored": (TARGETS, list(primary_weights)),
    "assay": (ASSAY_TARGETS, [assay_weight] * len(ASSAY_TARGETS)),
    "public": (public_targets, [public_weight] * len(public_targets)),
}
measured = {name: gradient_share(cols, ws) for name, (cols, ws) in groups.items()}
total_mass = sum(mass for _, mass in measured.values())

print(f"Building {model_name} on all {len(df):,} rows — no holdout")
print(f"pIC50 weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in primary_weights]))}")
print(f"assay weight:  {assay_weight:.3f} each ({ASSAY_WEIGHT} x mean primary), {len(ASSAY_TARGETS)} tasks")
print(f"public weight: {public_weight:.3f} each ({args.public_weight} x mean primary), {len(public_targets)} tasks")
if args.tox21:
    labelled = {t.split("_")[0]: int(df[t].notna().sum()) for t in TOX21_TARGETS}
    print(f"tox21 tasks: {labelled}")
print("gradient share (label count x weight, which is how chemprop reduces the loss):")
for name, (cols, _) in groups.items():
    labels, mass = measured[name]
    print(f"    {name:8s} {len(cols):3d} tasks {labels:9,} labels  {100 * mass / total_mass:5.1f}%")
if args.censored:
    flags = [f"{t}_lt" for t in PUBLIC_TARGETS if f"{t}_lt" in df.columns]
    counts = {c.replace("_pic50_chembl_lt", ""): int(df[c].fillna(False).astype(bool).sum()) for c in flags}
    print(f"chembl bounds: {counts}  (bounded_loss={bounded_loss})")
    if not flags:
        raise ValueError(f"{CENSORED_FS} carries no _lt columns — rebuild it with cyp_union_features.py --censored")

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=all_targets,
    description=f"Multi-task Chemprop, challenge + public targets, public weight {args.public_weight}"
    + (", ChEMBL bounds honoured" if bounded_loss else ", ChEMBL bounds as labels" if args.censored else ""),
    tags=TAGS + [f"public_weight_{args.public_weight}"] + (["censored"] if args.censored else []),
    hyperparameters={"task_weights": task_weights, "uq_version": "v1", "bounded_loss": bounded_loss},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()

# Out-of-fold over every row. The scored targets exist on 4,905 of them, which is the
# subset the variants are compared on.
end.cross_fold_inference()
