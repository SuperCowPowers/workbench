"""PXR phase-1 model: Chemprop with MVE on the shared FeatureSet (openadmet_pxr_f1).

Same held-out setup as pxr_chemprop_phase1.py (zero-weights the `phase1_test`
rows so Analog Set 1 never trains the model, then captures predictions on exactly
those rows), but the Chemprop FFN uses an MVE (mean-variance estimation) head:
each prediction also carries a per-molecule aleatoric variance, fed to UQ v1.

Because phase1_test has known pEC50 labels, the capture ('pxr_phase1_mve_test')
gives real held-out ground truth for evaluating MVE against the point-predictor
model (pxr-reg-chemprop-phase1 -> capture 'pxr_phase1_test'): point-accuracy
parity plus whether the aleatoric signal improves UQ calibration/sharpness.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Endpoint, ModelType, ModelFramework

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-mve-phase1"
tags = ["openadmet_pxr", "chemprop", "mve", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column="pec50",
    description="PXR phase-1 pEC50 Chemprop MVE (SMILES only; aleatoric UQ; phase1_test zero-weighted out)",
    tags=tags,
    # MVE head on; active confidence = v1 (uses the aleatoric feature). v0/v2 also saved.
    hyperparameters={"uq_version": "v1", "mve": True},
    sample_weights={mid: 0.0 for mid in phase1["molecule_name"]},  # held-out rows don't train
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
Endpoint(model_name).inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_mve_test")
