"""PXR phase-1 Chemprop + primary-screen readout features (the TabICL-spike control).

Same as pxr_chemprop_phase1.py, plus the two predicted log2FC readouts as extra
descriptors (openadmet_pxr_readout). TabICL gets the same readouts in
pxr_tabicl_spike_phase1.py, so this separates "the screen data helps" from
"TabICL helps". Holds phase1_test out via validation_ids and captures
'pxr_phase1_test' on exactly those rows.

Build the FeatureSet first:  ml_pipeline_launcher pxr_readout_feature_sets
"""

from workbench.api import FeatureSet, ModelType, ModelFramework

fs_name = "openadmet_pxr_readout"
model_name = "pxr-reg-chemprop-readout-phase1"
tags = ["openadmet_pxr", "chemprop", "primary_screen", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]
features = ["smiles", "lfc_8um_readout", "lfc_33um_readout"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=features,
    target_column="pec50",
    description="PXR phase-1 pEC50 Chemprop + primary-screen log2FC readouts (phase1_test held out)",
    tags=tags,
    validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
end.inference(phase1[features + ["molecule_name", "pec50"]], capture_name="pxr_phase1_test")
