"""Single-target Chemprop, one model per isoform — the ST arm of the model report.

One encoder per isoform, fit on that isoform's pIC50 and nothing else. Reads the same
FeatureSet as `cyp_chemprop_mt_aux_100.py`, so an ST-vs-MT comparison isolates the
multi-task structure: the MT model reaches every row through its other heads, where an ST
model sees only the rows its own target labels.

Rows with no label for the target go to `exclude_ids` rather than leaning on chemprop's
loss mask, so the fold split and the out-of-fold capture cover labelled rows only.

`--foundation` swaps the from-scratch encoder for CheMeleon, a D-MPNN pretrained on Mordred
descriptors over 1M PubChem molecules, and `--freeze-epochs` holds it fixed before fine-tuning
it. At one to two thousand labelled rows per isoform, fine-tuning a large pretrained encoder
end to end overfits, so the frozen-head warmup is the default.

`--width-weights` weights each row by the inverse of its label's credible-interval width. The
challenge scores a soft-threshold error that costs nothing for a prediction anywhere inside a
compound's interval, so a wide interval is cheap to miss. Three quarters of CYP2D6 labels are
tighter than 0.36 log units and a tenth span one to three; that tail currently trains the
encoder toward point estimates the metric ignores.

Chemprop serves from CPU, so these are serverless endpoints.

    ml_pipeline_launcher cyp_chemprop_st.py
    ml_pipeline_launcher cyp_chemprop_st.py -- --foundation
    ml_pipeline_launcher cyp_chemprop_st.py -- --isoforms cyp2d6
    ml_pipeline_launcher cyp_chemprop_st.py -- --isoforms cyp2d6 --width-weights
"""

import argparse

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_aux_f1"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
BASE_TAGS = ["openadmet_cyp", "chemprop", "single_task", "activity"]

# Bounds on an interval-width weight, so neither a near-zero interval nor a three-log-unit one
# dominates the loss. A compound at twice the median width counts half, and the clip costs about
# 12% of effective sample size on CYP2D6.
WEIGHT_CLIP = (0.5, 2.0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--isoforms",
    nargs="+",
    default=ISOFORMS,
    choices=ISOFORMS,
    help="Isoforms to build; defaults to all four",
)
parser.add_argument(
    "--name-suffix",
    default=None,
    help="Append to the model name, so a rebuild lands beside the existing model rather than replacing it. "
    "A retrain redraws the fold split, so its out-of-fold numbers move within seed noise",
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
    "--width-weights",
    action="store_true",
    help="Weight each row by the inverse of its label credible-interval width, clipped and "
    "normalized to mean 1. Lands beside the unweighted model as -widthw",
)
parser.add_argument(
    "--split-strategy",
    default="scaffold",
    choices=["scaffold", "butina", "random"],
    help="Fold grouping. Butina clusters Morgan fingerprints, holding analogue series whole",
)
parser.add_argument(
    "--butina-cutoff",
    type=float,
    default=0.6,
    help="Tanimoto DISTANCE cutoff for butina folds: 0.6 clusters at similarity >= 0.4",
)
args = parser.parse_args()
if args.freeze_epochs < 0:
    parser.error("--freeze-epochs cannot be negative")
if args.freeze_epochs != parser.get_default("freeze_epochs") and not args.foundation:
    parser.error("--freeze-epochs only applies with --foundation")

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

for iso in args.isoforms:
    target = f"{iso}_pic50_direct_inhibition"
    name = f"cyp-reg-chemprop-st-{iso.removeprefix('cyp')}"
    if args.name_suffix:
        name += f"-{args.name_suffix.strip('-')}"
    if args.foundation:
        name += f"-chemeleon-fz{args.freeze_epochs}"
    if args.width_weights:
        name += "-widthw"
    tags = BASE_TAGS + [iso] + (["chemeleon"] if args.foundation else [])
    if args.width_weights:
        tags.append("width_weights")

    hyperparameters = {
        "split_strategy": args.split_strategy,
        "butina_cutoff": args.butina_cutoff,
    }
    if args.foundation:
        hyperparameters["from_foundation"] = "CheMeleon"
        hyperparameters["freeze_mpnn_epochs"] = args.freeze_epochs

    exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
    print(f"{iso}: {len(df) - len(exclude_ids):,} labelled rows, excluding {len(exclude_ids):,} unlabelled")

    sample_weights = None
    if args.width_weights:
        lower, upper = f"{target}_ci_lower", f"{target}_ci_upper"
        measured = df[target].notna() & df[lower].notna() & df[upper].notna()
        width = (df.loc[measured, upper] - df.loc[measured, lower]).clip(lower=1e-3)
        w = (width.median() / width).clip(*WEIGHT_CLIP)
        w /= w.mean()
        sample_weights = dict(zip(df.loc[measured, "molecule_name"], w))
        print(
            f"{iso}: interval widths median {width.median():.2f}, weights {w.min():.2f}-{w.max():.2f}, "
            f"effective n {w.sum() ** 2 / (w**2).sum():.0f} of {len(w):,}"
        )

    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        feature_list=["smiles"],
        target_column=target,
        description=(
            f"CYP {iso.upper()} pIC50, single-target Chemprop"
            + (" from the CheMeleon pretrained encoder" if args.foundation else "")
        ),
        tags=tags,
        exclude_ids=exclude_ids,
        sample_weights=sample_weights,
        hyperparameters=hyperparameters,
    )
    model.set_owner("open_admet_cyp")

    end = model.to_endpoint(tags=tags)
    end.set_owner("open_admet_cyp")
    end.test_inference()
    end.cross_fold_inference()
