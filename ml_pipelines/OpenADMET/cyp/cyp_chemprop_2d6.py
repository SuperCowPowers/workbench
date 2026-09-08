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

`--foundation` swaps the from-scratch encoder for CheMeleon, a D-MPNN pretrained on Mordred
descriptors over 1M PubChem molecules. Four mechanisms have now failed to move the sub-4.0
band -- auxiliary heads, loss weighting, pooled public labels, a second assay arm -- and the
last one matched the direct model everywhere else while failing there, which points at the
representation rather than the target or the loss. CheMeleon's own headline is a 97% win rate
on MoleculeACE, an activity-cliff benchmark: similar structures, different potency, which is
the discrimination we lack.

`--freeze-epochs` holds the pretrained encoder fixed before fine-tuning it. At 1,493 rows
fine-tuning a large pretrained D-MPNN end to end will overfit, so read the two points
together:

    --foundation --freeze-epochs 0     fine-tune throughout
    --foundation --freeze-epochs 10    fit the head first, then adapt the encoder

`--scope tdi` trains on the +NADPH arm instead of the scored one and is then graded against
the scored labels, which sounds perverse and is the point. The two arms are separate curve
fits of the same molecules, and the TDI arm's fitted pIC50 ranks the scored labels below 4.0
at Spearman 0.679 where log2fc manages 0.037 -- a full dose-response keeps resolving potency
where a single-concentration readout has run out of range. That 0.679 is agreement between
two measurements, not something reachable from structure; the union model's TDI head, one of
26 at auxiliary weight 0.30, recovers 0.194 of it. What makes it worth a build anyway is that
0.194 still ranks the sub-4.0 scored labels better than the direct model's own -0.011, from a
head nobody optimised. The plausible mechanism is dynamic range: a head fit to the arm that
still varies down there cannot shrink the low end away.

Read `cyp_compare.py --bands`, which grades any CYP2D6 model against the scored labels
whatever column it trained on. Only the band Spearmans mean anything here -- the arms sit on
different scales, so ST-RAE and MAE do not transfer.

`--low-weight` and `--deep-weight` re-weight the low band via `sample_weights`, which chemprop
applies per datapoint: the 479 rows under pIC50 4.5, and the 129 under 4.0 separately. Both
degrade CYP2D6, monotonically in the share of the loss the 4.0-4.5 rows take -- 0.388
unweighted, 0.302 at `--low-weight 3`, 0.056 at 8, where the model is a flat line at the mean.
Those 350 labels spread 0.113 against a measurement std of 0.069, so there is no ordering in
them to learn, and a loss they dominate is minimized by a constant. The flags are kept for
re-measurement, not because a setting of them is expected to win.

A step rather than importance weights, deliberately. Weighting by `p_blind(y)/p_train(y)` is
the principled correction for the shift, and it fails twice here: uncapped it still only
reaches a weighted mean of 3.67 against the blind 3.107, because no reweighting reaches a
range with no samples in it, and it drops the effective sample size to 236 of 1,493. It also
aims at the wrong target -- matching the blind mean is a *level* correction, and placement
already does level. What is missing is ordering inside the band.

`--scope pooled` puts the public CYP2D6 measurements *in the scored column* rather than in
heads of their own. The scored head currently trains on 1,493 challenge rows while roughly
35,000 CYP2D6 measurements sit beside it as auxiliaries -- and a head keeps its own scale, so
none of that reaches the output we are graded on. Pooling is the only route that does.

`cyp_union_features.py` builds that column -- each source shifted onto the challenge scale by
the offset measured on the compounds both assays ran, averaged where sources overlap -- and
this script trains on it. Pooling belongs in the FeatureSet because it is data, not an
experiment knob; the knob is which column `target_column` names.

The objection is that the offset is measured on potent compounds and applied to weak ones, so
the corrected values are wrong at the low end. True, and it does not matter for the metric we
are losing: Spearman is rank-based, an offset that is roughly right preserves ordering, and
placement handles the absolute scale afterwards.

Grade this against `single` on the 1,493 challenge-labelled rows only. Its own out-of-fold set
now contains ~9,800 public rows, and a Spearman over those is not the number the board reads.

    --scope pooled                     challenge + public in one column, public at 1.0
    --scope pooled --public-weight 0.3 the same, public rows down-weighted

Public rows carry a residual of 0.35-0.50 after the shift against the challenge's own 0.07
label noise, so `--public-weight` is the knob for how much that noise is allowed to count.

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
# Built by cyp_union_features.py: the challenge labels with public potency shifted onto the
# same scale filling the gaps, plus a flag marking which rows came from the fill.
POOLED_TARGET = "cyp2d6_pic50_pooled"
# The +NADPH arm: the same molecules through the same dose-response design, fitted separately.
TDI_TARGET = "cyp2d6_pic50_tdi_condition"
POOLED_FLAG = "cyp2d6_pooled_public"
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
    choices=["single", "isoform", "pooled", "tdi"],
    help="'single' trains on the scored target alone; 'isoform' adds every CYP2D6 readout; "
    "'pooled' puts offset-corrected public measurements into the scored column; "
    "'tdi' trains on the TDI arm instead, to be scored against the direct labels",
)
parser.add_argument(
    "--foundation",
    action="store_true",
    help="Start from the CheMeleon pretrained encoder instead of training one from scratch",
)
parser.add_argument(
    "--freeze-epochs",
    type=int,
    default=10,
    help="Epochs to hold the pretrained encoder frozen before fine-tuning it (--foundation)",
)
parser.add_argument(
    "--public-weight",
    type=float,
    default=1.0,
    help="Loss weight for pooled public rows (--scope pooled). 1.0 trusts them like challenge rows",
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
if args.freeze_epochs < 0:
    parser.error("--freeze-epochs cannot be negative")
if args.freeze_epochs != parser.get_default("freeze_epochs") and not args.foundation:
    parser.error("--freeze-epochs only applies with --foundation")
if args.low_weight <= 0:
    parser.error("--low-weight must be positive")
if args.deep_weight is not None and args.deep_weight <= 0:
    parser.error("--deep-weight must be positive")
deep_weight = args.low_weight if args.deep_weight is None else args.deep_weight
weighted = args.low_weight != 1.0 or deep_weight != 1.0

model_name = f"cyp-reg-chemprop-2d6-{args.scope}"
if args.foundation:
    model_name += f"-chemeleon-fz{args.freeze_epochs}"
if args.scope == "pooled" and args.public_weight != 1.0:
    model_name += f"-pw{args.public_weight:g}".replace(".", "p")
if weighted:
    model_name += f"-lw{args.low_weight:g}".replace(".", "p")
    if deep_weight != args.low_weight:
        model_name += f"-dw{deep_weight:g}".replace(".", "p")
if args.scope == "pooled":
    targets = [POOLED_TARGET]
elif args.scope == "single":
    targets = [TARGET]
elif args.scope == "tdi":
    targets = [TDI_TARGET]
else:
    targets = [TARGET] + ISOFORM_AUX

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

pooled_rows = None
if args.scope == "pooled":
    missing = [c for c in (POOLED_TARGET, POOLED_FLAG) if c not in df.columns]
    if missing:
        raise ValueError(f"{FS_NAME} has no {missing} — rebuild it with cyp_union_features.py")
    pooled_rows = df[POOLED_FLAG].fillna(False).astype(bool)
    scored, both = df[TARGET].notna(), df[POOLED_TARGET].notna()
    print(
        f"Pooled scored column: {int(both.sum()):,} rows "
        f"({int(scored.sum()):,} challenge, {int(pooled_rows.sum()):,} public)"
    )
    for label, lo, hi in (("<4.0", -np.inf, 4.0), ("4.0-4.5", 4.0, 4.5), (">=4.5", 4.5, np.inf)):
        band = df[POOLED_TARGET].between(lo, hi, inclusive="left")
        print(
            f"  {label:>9s}  {int((band & scored).sum()):>6,} challenge  "
            f"{int((band & pooled_rows).sum()):>6,} public"
        )

trainable = int(df[targets].notna().any(axis=1).sum())
print(f"Building {model_name}: {len(targets)} target(s), {trainable:,} trainable rows of {len(df):,}")
for t in targets:
    print(f"  {t:<45}{int(df[t].notna().sum()):>7,}")

hyperparameters = {"uq_version": "v1"}
if args.foundation:
    hyperparameters["from_foundation"] = "CheMeleon"
    hyperparameters["freeze_mpnn_epochs"] = args.freeze_epochs
if len(targets) > 1:
    hyperparameters["task_weights"] = [1.0] + [AUX_WEIGHT] * len(ISOFORM_AUX)

# chemprop weights a datapoint, not a target, so on the isoform scope this reweights the
# compound across all its CYP2D6 readouts. That is the intent -- they are the same molecule
# being under-attended -- but it is why the single scope is the cleaner first read.
row_weight = None
if pooled_rows is not None and args.public_weight != 1.0:
    row_weight = np.where(pooled_rows, args.public_weight, 1.0)
    print(f"Pooled public rows weighted {args.public_weight:g}x against challenge rows")

sample_weights = None
if weighted:
    y = df[targets[0]]
    deep = y.notna() & (y < DEEP_BAND)
    mid = y.notna() & (y >= DEEP_BAND) & (y < LOW_BAND)
    band_weight = np.where(deep, deep_weight, np.where(mid, args.low_weight, 1.0))
    row_weight = band_weight if row_weight is None else row_weight * band_weight
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

if sample_weights is None and row_weight is not None:
    sample_weights = {mol: float(w) for mol, w in zip(df["molecule_name"], row_weight) if w != 1.0}

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
