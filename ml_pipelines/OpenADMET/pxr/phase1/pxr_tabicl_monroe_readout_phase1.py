"""PXR phase-1 TabICL on the Monroe embedding + primary-screen readout features.

The deployed version of pxr_monroe_spike_phase1.py's best arm: TabICL on the Monroe
embedding (the compressed feature `monroe`, which the template expands) plus the two
predicted log2FC readouts (openadmet_pxr_readout_monroe). TabICL keeps the 100
highest-variance embedding columns (top_variance_features): in
pxr_monroe_ablation_phase1.py, 100 scored RAE 0.537, 360 scored 0.540, and all 720 0.550,
differences within the noise of 253 compounds; 100 sits inside TabICL's pretraining range
and needs the least serving memory.
Holds phase1_test out via validation_ids and captures 'pxr_phase1_test' on exactly
those rows, for comparison against pxr-reg-chemprop-phase1 and
pxr-reg-chemprop-readout-phase1.

The endpoint is serverless: TabICL holds its training rows in memory, and with 100
embedding columns the training job measures 5.11 GB to serve, under the 5.5 GB serverless
limit. to_endpoint() raises if a retrain measures over it.

Build the FeatureSet first:  ml_pipeline_launcher pxr_readout_monroe_feature_sets
"""

from workbench.api import FeatureSet, ModelType, ModelFramework

fs_name = "openadmet_pxr_readout_monroe"
model_name = "pxr-reg-tabicl-monroe-readout-phase1"
tags = ["openadmet_pxr", "tabicl", "monroe", "primary_screen", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]
features = ["monroe", "lfc_8um_readout", "lfc_33um_readout"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.TABICL,
    feature_list=features,
    target_column="pec50",
    description="PXR phase-1 pEC50 TabICL on Monroe embedding + primary-screen log2FC readouts (phase1_test held out)",
    tags=tags,
    validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
    hyperparameters={"top_variance_features": 100},
)
m.set_owner("open_admet_pxr")
print(f"Measured serving memory: {(m.workbench_meta() or {}).get('workbench_inference_memory_gb')} GB")

end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
end.inference(phase1[features + ["molecule_name", "pec50"]], capture_name="pxr_phase1_test")
