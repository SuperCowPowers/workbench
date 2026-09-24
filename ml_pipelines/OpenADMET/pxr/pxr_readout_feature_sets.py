"""Producer: PXR pEC50 FeatureSet + primary-screen readout features.

openadmet_pxr_f1's rows (molecule_name, smiles, pec50, split) plus the readout
model's (pxr-reg-chemprop-lfc) predicted log2FC at 8.25 and 33 uM:

  - compounds in the screen: out-of-fold predictions, so a pEC50 model never sees a
    readout the screen model was fit on (these are all `train` rows)
  - all other compounds (incl. every phase1_test row): the endpoint's predictions

Consumed by phase1/pxr_chemprop_readout_phase1.py and phase1/pxr_tabicl_spike_phase1.py.

Run after the readout model:  ml_pipeline_launcher pxr_chemprop_lfc
"""

from workbench.api import DataSource, Endpoint, FeatureSet, Model

READOUT_FS = "openadmet_pxr_readout"
readout_model = "pxr-reg-chemprop-lfc"
READOUTS = {"lfc_8um_pred": "lfc_8um_readout", "lfc_33um_pred": "lfc_33um_readout"}

df = FeatureSet("openadmet_pxr_f1").pull_dataframe()[["molecule_name", "smiles", "pec50", "split"]]

# Endpoint predictions for every row, then OOF over the rows the screen model trained on
preds = Endpoint(readout_model).inference(df[["molecule_name", "smiles"]])
readout = preds.set_index("molecule_name")[list(READOUTS)]
oof = Model(readout_model).get_inference_predictions("model_training")
screen_smiles = FeatureSet("openadmet_pxr_lfc").pull_dataframe().set_index("ocnt_id")["smiles"]
oof = oof.assign(smiles=oof["ocnt_id"].map(screen_smiles)).set_index("smiles")[list(READOUTS)]
in_screen = df["smiles"].isin(oof.index)
assert not df.loc[in_screen, "split"].ne("train").any(), "phase1_test compound found in the primary screen"

for pred_col, feat_col in READOUTS.items():
    df[feat_col] = df["molecule_name"].map(readout[pred_col])
    df.loc[in_screen, feat_col] = df.loc[in_screen, "smiles"].map(oof[pred_col])
assert df[list(READOUTS.values())].notna().all().all(), "missing readout predictions"

DataSource(df, name=f"{READOUT_FS}_ds").to_features(
    READOUT_FS, id_column="molecule_name", tags=["openadmet_pxr", "primary_screen", "activity"]
)
print(f"Built '{READOUT_FS}': {len(df)} rows, {in_screen.sum()} with OOF readouts (in screen)")
