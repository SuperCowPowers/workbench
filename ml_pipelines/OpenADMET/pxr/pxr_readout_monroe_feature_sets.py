"""Producer: PXR pEC50 FeatureSet + primary-screen readouts + Monroe embedding.

openadmet_pxr_readout's rows (molecule_name, smiles, pec50, split, the two predicted
log2FC readouts) plus the 720-d Monroe embedding from the smiles-to-monroe-v1 feature
endpoint, as the compressed feature `monroe` (the model templates expand it).
Molecules Monroe cannot embed are left out.

Consumed by phase1/pxr_tabicl_monroe_readout_phase1.py.

Run after the readout FeatureSet:  ml_pipeline_launcher pxr_readout_feature_sets
"""

from workbench.api import DataSource, Endpoint, FeatureSet
from workbench.api.inference_cache import InferenceCache

MONROE_FS = "openadmet_pxr_readout_monroe"
READOUTS = ["lfc_8um_readout", "lfc_33um_readout"]

df = FeatureSet("openadmet_pxr_readout").pull_dataframe()[["molecule_name", "smiles", "pec50", "split"] + READOUTS]

# SMILES-keyed cache (S3-persisted), so a rebuild only embeds molecules it hasn't seen.
# A redeployed endpoint can change its output, so the cache resets when the endpoint does.
cached = InferenceCache(Endpoint("smiles-to-monroe-v1"), auto_invalidate_cache=True)
feat_df = cached.inference(df)

embedded = feat_df["monroe"].notna()
assert embedded[feat_df["split"] == "phase1_test"].all(), "Monroe failed to embed a phase1_test molecule"
feat_df = feat_df[embedded]

DataSource(feat_df, name=f"{MONROE_FS}_ds").to_features(
    MONROE_FS, id_column="molecule_name", tags=["openadmet_pxr", "primary_screen", "monroe", "activity"]
)
FeatureSet(MONROE_FS).set_compressed_features(["monroe"])
print(
    f"Built '{MONROE_FS}': {len(feat_df)} of {len(df)} rows embedded "
    f"({(feat_df.split == 'train').sum()} train + {(feat_df.split == 'phase1_test').sum()} phase1_test)"
)
