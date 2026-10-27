"""CYP Atom-JEPA on 3D conformers — one model per isoform.

A fourth encoder family for the pool. Atom-JEPA is an EquiformerV3 (39M params) pretrained
by latent prediction over ~19M Uni-Mol molecules; this fine-tunes it, which is where the
paper's ADMET wins come from -- its frozen embedding trails Monroe's in the same position.

Its value here is decorrelation rather than score. Our current members agree at rho 0.73-0.97
and leave-one-out shows most of them contributing almost nothing, which is the measured gap
against a field that ensembles a dozen genuinely different models. A graph-free 3D encoder is
the furthest thing from a D-MPNN we have.

**Gate on solo strength before pooling.** The combination is a flat mean, so a weak member
carries the same vote as a strong one, and the board has refuted our ability to down-weight
it. Compare against the isoform's best member with `scripts/cyp_compare.py` and include only
within the resolution threshold -- 0.033 / 0.028 / 0.037 / 0.012 paired, since these share a
row set and seed with the TabICL and single-target chemprop arms and therefore share folds.

`--readouts` trains on the screen-readout FeatureSet, feeding the four predicted log2FC
columns to the readout head alongside the conformers. Those features put TabICL first on all
four isoforms, so this is the arm to build first.

Serves from a GPU image: `async_endpoint=True` scales to zero when idle, where serverless is
rejected outright. Every isoform trains before any endpoint is deployed, so a GPU capacity
miss costs an endpoint rather than the remaining isoforms' training; `cyp_atomjepa_deploy.py`
picks up whatever it stranded.

    ml_pipeline_launcher cyp_atomjepa.py -- --readouts
    ml_pipeline_launcher cyp_atomjepa.py -- --isoforms cyp2d6
"""

import argparse

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_aux_f1"
READOUT_FS = "openadmet_cyp_readout_monroe_f1"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
BASE_TAGS = ["openadmet_cyp", "atomjepa", "activity"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--readouts",
    action="store_true",
    help="Feed the four predicted log2FC columns to the readout head",
)
parser.add_argument(
    "--name-suffix",
    default=None,
    help="Append to the model name, so a rebuild lands beside the existing model rather than replacing it. "
    "A retrain redraws the fold split, so its out-of-fold numbers move within seed noise",
)
parser.add_argument(
    "--isoforms",
    nargs="+",
    default=ISOFORMS,
    choices=ISOFORMS,
    help="Isoforms to build; defaults to all four",
)
args = parser.parse_args()

fs = FeatureSet(READOUT_FS if args.readouts else FS_NAME)
df = fs.pull_dataframe()
readouts = [f"{iso}_log2fc_readout" for iso in ISOFORMS] if args.readouts else []

trained = []
for iso in args.isoforms:
    target = f"{iso}_pic50_direct_inhibition"
    name = f"cyp-reg-atomjepa-{iso.removeprefix('cyp')}"
    if args.readouts:
        name += "-readout"
    if args.name_suffix:
        name += f"-{args.name_suffix.strip('-')}"
    tags = BASE_TAGS + [iso]

    # Nothing drops NaN-target rows on the way into a training view, and coverage is sparse,
    # so the unlabelled rows go to exclude_ids. Chemprop gets away without this because its
    # multi-task loss masks missing targets.
    exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
    print(f"{iso}: {len(df) - len(exclude_ids):,} labelled rows, excluding {len(exclude_ids):,} unlabelled")

    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.ATOMJEPA,
        feature_list=["smiles"] + readouts,
        target_column=target,
        description=f"CYP {iso.upper()} pIC50, Atom-JEPA fine-tuned over 3D conformers",
        tags=tags,
        exclude_ids=exclude_ids,
    )
    model.set_owner("open_admet_cyp")
    trained.append((model, tags))

for model, tags in trained:
    # An A10G (ml.g5.xlarge): the default ml.g6.xlarge often lacks capacity in us-west-2
    end = model.to_endpoint(tags=tags, async_endpoint=True, instance="ml.g5.xlarge")
    end.set_owner("open_admet_cyp")
    end.test_inference()
    end.cross_fold_inference()
