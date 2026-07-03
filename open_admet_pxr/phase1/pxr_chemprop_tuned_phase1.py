"""PXR phase-1 model: regularization-tuned ChemProp on the shared FeatureSet.

Same held-out setup as pxr_chemprop_phase1.py (the phase1_test rows are
zero-weighted out of training and captured), but with the ChemProp knobs tuned
toward regularization for this ~4k-compound dataset: a tapered FFN head (far
fewer params than the default 2000x2), slightly higher dropout, and a gentler
learning-rate schedule. Compares directly against the default-knob
pxr-reg-chemprop-phase1 on the same held-out rows.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Endpoint, ModelType, ModelFramework

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-tuned-phase1"
tags = ["openadmet_pxr", "chemprop", "tuned", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column="pec50",
    description="PXR phase-1 pEC50 Chemprop, regularization-tuned (tapered FFN + more dropout + gentler LR)",
    tags=tags,
    hyperparameters={
        "uq_version": "v1",
        "ffn_hidden_dim": [1024, 256, 64],  # tapered head (vs default 2000x2)
        "dropout": 0.2,                      # vs default 0.1
        "warmup_epochs": 5,                  # gentler schedule
        "max_lr": 5e-4,                      # vs default 1e-3
    },
    sample_weights={mid: 0.0 for mid in phase1["molecule_name"]},  # held-out rows don't train
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
Endpoint(model_name).inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_tuned_test")
