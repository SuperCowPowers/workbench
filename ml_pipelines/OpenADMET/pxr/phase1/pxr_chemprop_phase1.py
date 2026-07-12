"""PXR phase-1 model: Chemprop on the shared FeatureSet (openadmet_pxr_f1).

Zero-weights the `phase1_test` rows via sample_weights so the held-out Analog
Set 1 never trains the model, then captures predictions on exactly those rows as
'pxr_phase1_test'. Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Endpoint, ModelType, ModelFramework

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-phase1"
tags = ["openadmet_pxr", "chemprop", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column="pec50",
    description="PXR phase-1 pEC50 Chemprop (SMILES only; phase1_test zero-weighted out of training)",
    tags=tags,
    hyperparameters={"uq_version": "v1"},  # active confidence = v1; v0/v2 also saved
    sample_weights={mid: 0.0 for mid in phase1["molecule_name"]},  # held-out rows don't train
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
Endpoint(model_name).inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
