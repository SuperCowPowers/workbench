"""Producer: the Monroe embedding plus predicted log2FC readouts.

`openadmet_cyp_monroe_f1`'s rows and labels, plus `cyp-reg-chemprop-log2fc`'s predicted
log2FC for all four isoforms as features:

  - compounds the screen measured: out-of-fold predictions, so a pIC50 model never reads a
    readout the screen model was fit on
  - every other compound: the endpoint's predictions, which are already out-of-sample for
    rows that model never trained on

Out of fold, against the same Monroe columns alone (`scripts/cyp_readout_spike.py`, shared
scaffold folds, paired over compounds):

    isoform   monroe   + readouts   paired delta            resolves at
    CYP1A2    0.5482       0.5817   +0.0337 [+0.017,+0.049]      0.043
    CYP2C9    0.6757       0.6984   +0.0225 [+0.007,+0.036]      0.031
    CYP2D6    0.4471       0.4664   +0.0194 [+0.007,+0.030]      0.056
    CYP3A4    0.7784       0.8051   +0.0267 [+0.018,+0.036]      0.018

CYP3A4 clears its threshold; the rest sit under theirs with paired intervals excluding zero.

The screen is disjoint from the blind set (0 of 748 skeletons), so the 750 take endpoint
readouts at submission time, as the Monroe embedding already does.

Run after the readout model:  ml_pipeline_launcher cyp_chemprop_log2fc
"""

from workbench.api import DataSource, Endpoint, FeatureSet, Model

SOURCE_FS = "openadmet_cyp_monroe_f1"
READOUT_FS = "openadmet_cyp_readout_monroe_f1"
READOUT_MODEL = "cyp-reg-chemprop-log2fc"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
READOUTS = {f"{iso}_log2fc": f"{iso}_log2fc_readout" for iso in ISOFORMS}

df = FeatureSet(SOURCE_FS).pull_dataframe()

# Endpoint predictions for every row, then out-of-fold over the rows the screen model trained on.
preds = Endpoint(READOUT_MODEL).inference(df[["molecule_name", "smiles"]].copy()).set_index("molecule_name")
model = Model(READOUT_MODEL)
for target, feature in READOUTS.items():
    df[feature] = df["molecule_name"].map(preds[f"{target}_pred"])
    oof = model.get_inference_predictions(f"cv_{target}").set_index("molecule_name")["prediction"]
    in_screen = df["molecule_name"].isin(oof.index)
    df.loc[in_screen, feature] = df.loc[in_screen, "molecule_name"].map(oof)
    print(f"{feature}: {int(in_screen.sum()):,} out-of-fold, {int((~in_screen).sum()):,} from the endpoint")

missing = df[list(READOUTS.values())].isna().any(axis=1)
if missing.any():
    raise ValueError(f"{int(missing.sum())} rows have no readout; every row must carry all four")

DataSource(df, name=f"{READOUT_FS}_ds").to_features(
    READOUT_FS, id_column="molecule_name", tags=["openadmet_cyp", "monroe", "readout", "activity"]
)
FeatureSet(READOUT_FS).set_compressed_features(["monroe"])
print(f"Built '{READOUT_FS}': {len(df):,} rows, monroe + {len(READOUTS)} readouts")
