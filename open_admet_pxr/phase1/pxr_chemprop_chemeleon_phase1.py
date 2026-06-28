"""PXR phase-1 CheMeleon freeze sweep — pick freeze_mpnn_epochs on held-out RAE.

Chemprop warm-started from the CheMeleon foundation model. CheMeleon's validated
protocol is full fine-tune (freeze=0); a short freeze (linear-probe-then-
fine-tune, Kumar et al. 2022) is theoretically better for our small + OOD regime
but unproven here — so we sweep it rather than guess. For each freeze value a
model is built, the phase1_test rows are zero-weighted out of training, and a
'pxr_phase1_test' capture is run on exactly those rows for an honest held-out
comparison. Lock the phase-2 submission model to the winner.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Model, Endpoint, ModelType, ModelFramework

recreate = False
fs_name = "openadmet_pxr_f1"
FREEZE_EPOCHS = [0, 10, 20]  # 0 = CheMeleon's validated full fine-tune; >0 = LP-FT bet for OOD
base_tags = ["openadmet_pxr", "chemprop", "chemeleon", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]
weights = {mid: 0.0 for mid in phase1["molecule_name"]}  # held-out rows don't train

for frz in FREEZE_EPOCHS:
    model_name = f"pxr-reg-chemprop-chemeleon-phase1-frz{frz}"
    tags = base_tags + [f"frz{frz}"]
    if recreate or not Model(model_name).exists():
        m = fs.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            feature_list=["smiles"],
            target_column="pec50",
            description=f"PXR phase-1 Chemprop, CheMeleon warm-start, freeze={frz} (phase1_test zero-weighted)",
            tags=tags,
            hyperparameters={"uq_version": "v1", "from_foundation": "CheMeleon", "freeze_mpnn_epochs": frz},
            sample_weights=weights,
        )
        m.set_owner("open_admet_pxr")
        end = m.to_endpoint(tags=tags)
        end.set_owner("open_admet_pxr")
        end.test_inference()
        end.cross_fold_inference()
    # Held-out capture on the phase1_test rows (the model never trained on them)
    Endpoint(model_name).inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
