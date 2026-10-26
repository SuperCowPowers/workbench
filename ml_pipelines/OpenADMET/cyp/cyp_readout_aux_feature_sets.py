"""Producer: the Monroe embedding plus predicted auxiliary readouts.

`openadmet_cyp_monroe_f1`'s rows and labels, plus a readout model's predictions for every
auxiliary family it carries, as features.

A readout is a *feature*, not a head, which is what lets it carry information the auxiliary
heads never could -- a head keeps its own scale, so low-range information does not cross
between heads. With log2FC alone this puts TabICL first on all four isoforms out of fold.

**The readout model must carry no direct-inhibition head.** Its predictions feed a pIC50
model, so a pIC50 head anywhere in it would be in-sample for the target it feeds.
`cyp_chemprop_log2fc.py` enforces this; `cyp-reg-chemprop-union-p30` violates it and must
never be read out.

Each family resolves per compound as: out-of-fold predictions where the readout model was
fit on that row, so a pIC50 model never reads a readout fitted to it, and the endpoint's
prediction everywhere else, which is already out-of-sample for a row that model never saw.

    ml_pipeline_launcher cyp_readout_aux_feature_sets.py
    ml_pipeline_launcher cyp_readout_aux_feature_sets.py -- --model cyp-reg-chemprop-chembl-log2fc-max-response-veith

Run after the readout model:  ml_pipeline_launcher cyp_chemprop_log2fc.py -- --families log2fc veith chembl max_response
"""

import argparse

from workbench.api import DataSource, Endpoint, FeatureSet, Model

SOURCE_FS = "openadmet_cyp_monroe_f1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    default="cyp-reg-chemprop-log2fc",
    help="Readout model to score. Must carry no direct-inhibition head",
)
parser.add_argument(
    "--name",
    default=None,
    help="FeatureSet to write; defaults to a name derived from the readout model",
)
args = parser.parse_args()

model = Model(args.model)
targets = model.target()
targets = list(targets) if isinstance(targets, list) else [targets]

leaks = [t for t in targets if "direct_inhibition" in t]
if leaks:
    raise ValueError(
        f"{args.model} carries direct-inhibition heads {leaks}; its predictions would be "
        "in-sample for the target they feed. Use a readout model with no pIC50 head."
    )

fs_name = args.name or f"openadmet_cyp_readout_{args.model.removeprefix('cyp-reg-chemprop-').replace('-', '_')}_f1"
readouts = {t: f"{t}_readout" for t in targets}
print(f"Reading out {len(targets)} heads from {args.model} into '{fs_name}'")

df = FeatureSet(SOURCE_FS).pull_dataframe()

# Endpoint predictions for every row, then out-of-fold over the rows the readout model fit.
preds = Endpoint(args.model).inference(df[["molecule_name", "smiles"]].copy()).set_index("molecule_name")
for target, feature in readouts.items():
    df[feature] = df["molecule_name"].map(preds[f"{target}_pred"])
    oof = model.get_inference_predictions(f"cv_{target}")
    if oof is not None:
        oof = oof.set_index("molecule_name")["prediction"]
        in_fit = df["molecule_name"].isin(oof.index)
        df.loc[in_fit, feature] = df.loc[in_fit, "molecule_name"].map(oof)
        print(f"  {feature}: {int(in_fit.sum()):,} out-of-fold, {int((~in_fit).sum()):,} from the endpoint")
    else:
        print(f"  {feature}: no cv capture, all {len(df):,} from the endpoint")

missing = df[list(readouts.values())].isna().any(axis=1)
if missing.any():
    raise ValueError(f"{int(missing.sum())} rows have no readout; every row must carry all {len(readouts)}")

DataSource(df, name=f"{fs_name}_ds").to_features(
    fs_name, id_column="molecule_name", tags=["openadmet_cyp", "monroe", "readout", "activity"]
)
FeatureSet(fs_name).set_compressed_features(["monroe"])
print(f"Built '{fs_name}': {len(df):,} rows, monroe + {len(readouts)} readouts")
