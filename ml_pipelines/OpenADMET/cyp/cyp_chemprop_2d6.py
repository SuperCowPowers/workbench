"""CYP2D6 with its own encoder — does dropping the sharing do what re-weighting could not?

Re-weighting CYP2D6's head by 40x moved its ranking by 0.004 on an 8,415-row ruler.
A head with enough capacity fits its target whatever the loss
says, so CYP2D6's ordering is coming from the shared encoder, which every other target also
shapes. That leaves one structural question the weighting experiment could not reach:
whether sharing an encoder across isoforms is what holds CYP2D6 back.

The board hints that it might. Its two best CYP2D6 entries are poor on the other three
isoforms -- the profile of a specialist -- while the entries strong everywhere sit mid-pack
on CYP2D6. CYP2D6 is also the isoform the challenge did not select hits on, whose
single-concentration readout is flat where the others are monotone, and the only one where
emax carries potency signal. Several independent reasons to think it is not like the other
three.

Two scopes, both with an encoder that sees CYP2D6 and nothing else:

    --scope single    the scored target alone, 1,493 rows. Maximum isolation, minimum data.
    --scope isoform   every CYP2D6 readout we have, ~16k rows. Drops cross-isoform sharing
                      while keeping within-isoform signal, which is the specialist design
                      an actual competitor would build.

Compare against `cyp-reg-chemprop-union-p30` on `cv_cyp2d6_pic50_direct_inhibition`, against
the thresholds in `scripts/cyp_ruler_power.py` -- 0.056 on the target itself, 0.031 on
log2fc. The training row set differs between variants, so scaffold folds differ too; that is
inherent to the comparison rather than a flaw, but it is not seed-for-seed.

If neither scope moves CYP2D6, representation sharing is not the problem and the remaining
hypothesis is features -- which is where the XGB-on-descriptors tie points.

`--low-weight` is the separate experiment: not where CYP2D6's ordering comes from, but which
compounds the loss pays attention to. Out of fold this model ranks at Spearman 0.159 below
pIC50 4.5 against 0.383 above it, and the blind population is centred at 3.107 -- so it
cannot order the compounds it is mostly scored on. The flag multiplies the loss weight of
the 479 rows under 4.5, via `sample_weights`, which chemprop applies per datapoint.

    --low-weight 3    the low band becomes ~59% of the loss, effective sample size ~76%
    --low-weight 8    ~79% of the loss, effective sample size ~50%

`--deep-weight` splits that band in two, because its halves are not alike. Between 4.0 and
4.5 the 350 labels have a spread of 0.11 against a median measurement std of 0.069 -- they
are very nearly tied, and out of fold we rank them at 0.024, about what no signal looks like.
Below 4.0 the 129 labels spread 0.68 and we rank them at 0.153. A flat step spends most of
its weight on the half with nothing to order:

    --low-weight 3 --deep-weight 6    sub-4.0 takes 27% of the loss, ESS ~61%
    --low-weight 3 --deep-weight 9    sub-4.0 takes 36% of the loss, ESS ~48%

The 4.0-4.5 rows still earn a weight above 1 -- they have to sit below the actives, which is
cross-band ordering the scored Spearman does care about -- just not the majority share.

Two points bracket it, because a dose-response is worth more than either alone. Read
**low-band** Spearman, not overall: the overall number dilutes a large low-band change across
the 1,014 compounds already ranked well, and its 0.056 threshold would hide the effect.

What weighting cannot do is manufacture signal the labels do not carry. 129 compounds with
real spread and 350 that are one value with noise on top may simply not support an ordering.

A step rather than importance weights, deliberately. Weighting by `p_blind(y)/p_train(y)` is
the principled correction for the shift, and it fails twice here: uncapped it still only
reaches a weighted mean of 3.67 against the blind 3.107, because no reweighting reaches a
range with no samples in it, and it drops the effective sample size to 236 of 1,493. It also
aims at the wrong target -- matching the blind mean is a *level* correction, and placement
already does level. What is missing is ordering inside the band.

Build the FeatureSet first: python cyp_union_features.py
"""

import argparse

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_union_f1"
TAGS = ["openadmet_cyp", "chemprop", "activity", "cyp2d6_specialist"]

TARGET = "cyp2d6_pic50_direct_inhibition"
# Every other CYP2D6 readout: the challenge's own arms, then the public panels.
ISOFORM_AUX = [
    "cyp2d6_log2fc",
    "cyp2d6_pic50_tdi_condition",
    "cyp2d6_emax_vs_pos_ctrl_direct_inhibition",
    "cyp2d6_pic50_chembl",
    "cyp2d6_max_response",
]
AUX_WEIGHT = 0.3  # the value the auxiliary heads were validated at elsewhere
# Below this the model cannot order compounds (out-of-fold Spearman 0.159 against 0.383
# above). 4.5 rather than 4.0: the sub-4.0 set is 129 rows, too few to learn an ordering
# from, where 4.5 reaches 479.
LOW_BAND = 4.5
# Inside the low band, only below here do the labels carry spread worth ordering.
DEEP_BAND = 4.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scope",
    required=True,
    choices=["single", "isoform"],
    help="'single' trains on the scored target alone; 'isoform' adds every CYP2D6 readout",
)
parser.add_argument(
    "--low-weight",
    type=float,
    default=1.0,
    help=f"Loss weight for rows under pIC50 {LOW_BAND}; 1.0 (default) weights every row equally",
)
parser.add_argument(
    "--deep-weight",
    type=float,
    default=None,
    help=f"Loss weight for rows under pIC50 {DEEP_BAND}. Defaults to --low-weight, making the "
    f"band a flat step; raise it to favour the half of the band whose labels carry spread",
)
args = parser.parse_args()
if args.low_weight <= 0:
    parser.error("--low-weight must be positive")
if args.deep_weight is not None and args.deep_weight <= 0:
    parser.error("--deep-weight must be positive")
deep_weight = args.low_weight if args.deep_weight is None else args.deep_weight
weighted = args.low_weight != 1.0 or deep_weight != 1.0

model_name = f"cyp-reg-chemprop-2d6-{args.scope}"
if weighted:
    model_name += f"-lw{args.low_weight:g}".replace(".", "p")
    if deep_weight != args.low_weight:
        model_name += f"-dw{deep_weight:g}".replace(".", "p")
targets = [TARGET] if args.scope == "single" else [TARGET] + ISOFORM_AUX

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

trainable = int(df[targets].notna().any(axis=1).sum())
print(f"Building {model_name}: {len(targets)} target(s), {trainable:,} trainable rows of {len(df):,}")
for t in targets:
    print(f"  {t:<45}{int(df[t].notna().sum()):>7,}")

hyperparameters = {"uq_version": "v1"}
if len(targets) > 1:
    hyperparameters["task_weights"] = [1.0] + [AUX_WEIGHT] * len(ISOFORM_AUX)

# chemprop weights a datapoint, not a target, so on the isoform scope this reweights the
# compound across all its CYP2D6 readouts. That is the intent -- they are the same molecule
# being under-attended -- but it is why the single scope is the cleaner first read.
sample_weights = None
if weighted:
    y = df[TARGET]
    deep = y.notna() & (y < DEEP_BAND)
    mid = y.notna() & (y >= DEEP_BAND) & (y < LOW_BAND)
    row_weight = np.where(deep, deep_weight, np.where(mid, args.low_weight, 1.0))
    sample_weights = {mol: float(w) for mol, w in zip(df["molecule_name"], row_weight) if w != 1.0}

    # Diagnostics over the rows that actually train — on the isoform scope that is every
    # CYP2D6 readout, so the low band is a far smaller share of the loss than on `single`.
    trainable = df[targets].notna().any(axis=1).to_numpy()
    w = row_weight[trainable]
    print(
        f"Low-band weighting: under {DEEP_BAND} at {deep_weight:g}x, " f"{DEEP_BAND}-{LOW_BAND} at {args.low_weight:g}x"
    )
    for label, mask in (
        ("<" + str(DEEP_BAND), deep),
        (f"{DEEP_BAND}-{LOW_BAND}", mid),
        (">=" + str(LOW_BAND), ~(deep | mid)),
    ):
        m = mask.to_numpy() & trainable
        print(f"  {label:>9s}  {int(m.sum()):5,} rows  {100 * row_weight[m].sum() / w.sum():5.1f}% of the loss")
    ess = w.sum() ** 2 / (w**2).sum()
    print(f"  effective sample size {ess:,.0f} of {len(w):,} ({ess / len(w):.0%})")

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=targets,
    description=f"CYP2D6-only Chemprop, scope={args.scope}"
    + (f", low band {args.low_weight:g}x" if args.low_weight != 1.0 else ""),
    tags=TAGS + [args.scope],
    hyperparameters=hyperparameters,
    sample_weights=sample_weights,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()
