"""CYP TabICL on the Monroe embedding — one model per isoform.

A different representation, which is the class of change the CYP2D6 record had not yet
tried: TabICL over a 720-d molecular embedding rather than a D-MPNN over the graph.
Everything before it changed the target, the loss, or the data, and came back null.

Measured on CYP2D6, out of fold on its 1,493 labelled rows:

    model                      rho     AUC <4.0 vs >=4.5   corr with the 4-model pool
    tabicl-2d6-monroe        0.448                 0.799                        0.769
    chemprop-2d6-isoform     0.445                 0.763                        0.867
    chemprop-2d6-single      0.388                 0.706                        0.775

The AUC is the number that matters. CYP2D6's failure is telling a sub-4.0 compound from a
>=4.5 one -- a 3.5-log distinction carrying no label-noise excuse, where CYP3A4 reaches
0.939 -- and this is the first change to move it. The decorrelation is real too: the four
chemprops sit at 0.861-0.871 with each other.

`--top-variance` keeps that many of the 720 embedding columns, ranked by variance on the
training rows. 360 is the default because it is best or tied-best everywhere measured
(`scripts/cyp_monroe_ablation.py`, out of fold over shared scaffold folds):

    arm        CYP3A4 rho   CYP2D6 rho
    top100         0.7958       0.4516
    top200         0.8010       0.4523
    top360         0.8028       0.4536
    all720         0.8042       0.4521
    pca100         0.8000       0.4371

Paired against top100, CYP3A4 resolves at top360 (+0.0068, CI excluding zero) while CYP2D6 is
flat across every width. The gain tracks row count -- CYP3A4 carries 2,335 labelled rows
against CYP2D6's 1,493 -- so expect CYP1A2 and CYP2C9 to be flat as well. PCA at the same
width loses ~0.015 on CYP2D6 and gains nothing on CYP3A4; the template supports it via
`pca_components` and it is not worth reaching for.

More columns cost serving memory, which TabICL measures at training time: `to_endpoint()`
raises above the 5.5 GB serverless limit, and a model over it deploys with `serverless=False`.

`--readouts` trains on `openadmet_cyp_readout_monroe_f1` instead, adding the screen model's
four predicted log2FC columns as features. Measured out of fold, paired over compounds against
the same Monroe columns alone: CYP1A2 +0.0337, CYP3A4 +0.0267, CYP2C9 +0.0225, CYP2D6 +0.0194,
every interval excluding zero and CYP3A4 clearing its 0.018 threshold. A readout is a feature
rather than a head, which is what lets it carry information the auxiliary heads never could.

TabICL is single-target, so each isoform gets its own model. Multi-task is cheap to lose
here: `mt-aux-100` against `union-p30-h26` is -0.002 mean across four isoforms.

Nothing drops NaN-target rows on the way into a training view, and coverage is sparse --
1,285 to 2,335 labels in a 4,905-row set -- so the unlabelled rows go to `exclude_ids`.
Chemprop gets away without this because its multi-task loss masks missing targets.

Read with `scripts/cyp_compare.py --bands` against the per-isoform thresholds it prints.

    ml_pipeline_launcher cyp_tabicl_monroe
    ml_pipeline_launcher cyp_tabicl_monroe -- --isoforms cyp1a2 cyp2c9 cyp3a4
    ml_pipeline_launcher cyp_tabicl_monroe -- --top-variance 720 --name-suffix t720
    ml_pipeline_launcher cyp_tabicl_monroe -- --readouts

Build the FeatureSet first:  ml_pipeline_launcher cyp_monroe_feature_sets
"""

import argparse

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_monroe_f1"
READOUT_FS = "openadmet_cyp_readout_monroe_f1"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
BASE_TAGS = ["openadmet_cyp", "tabicl", "monroe", "activity"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--readouts",
    action="store_true",
    help="Train on the readout FeatureSet, adding the four predicted log2FC columns as features",
)
parser.add_argument(
    "--top-variance",
    type=int,
    default=360,
    help="Keep this many of the 720 Monroe columns, ranked by variance on the training rows",
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

for iso in args.isoforms:
    target = f"{iso}_pic50_direct_inhibition"
    name = f"cyp-reg-tabicl-{iso.removeprefix('cyp')}-monroe"
    if args.readouts:
        name += "-readout"
    if args.name_suffix:
        name += f"-{args.name_suffix.strip('-')}"
    tags = BASE_TAGS + [iso]

    exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
    print(f"{iso}: {len(df) - len(exclude_ids):,} labelled rows, excluding {len(exclude_ids):,} unlabelled")

    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.TABICL,
        feature_list=["monroe"] + readouts,
        target_column=target,
        description=f"CYP {iso.upper()} pIC50, TabICL over the Monroe embedding",
        tags=tags,
        hyperparameters={"top_variance_features": args.top_variance},
        exclude_ids=exclude_ids,
    )
    model.set_owner("open_admet_cyp")
    print(f"  serving memory: {(model.workbench_meta() or {}).get('workbench_inference_memory_gb')} GB")

    end = model.to_endpoint(tags=tags)
    end.set_owner("open_admet_cyp")
    end.test_inference()
    end.cross_fold_inference()
