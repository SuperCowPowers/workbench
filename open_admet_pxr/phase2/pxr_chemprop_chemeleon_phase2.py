"""PXR phase-2 model: Chemprop warm-started from the CheMeleon foundation model.

Same as pxr_chemprop_phase2.py but the MPNN is initialized from CheMeleon
pretrained weights (`from_foundation`). Trains on ALL rows (train + revealed
phase-1), predicts the 513-compound blinded phase-2 test set, and writes a
submission CSV. Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Model, Endpoint, ModelType, ModelFramework, PublicData

recreate = False
fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-chemeleon-phase2"
tags = ["openadmet_pxr", "chemprop", "chemeleon", "phase2"]

if recreate or not Model(model_name).exists():
    m = FeatureSet(fs_name).to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        feature_list=["smiles"],
        target_column="pec50",
        description="PXR phase-2 pEC50 Chemprop, CheMeleon foundation warm-start (trains on train + phase-1)",
        tags=tags,
        # CheMeleon foundation warm-start. TODO: set freeze_mpnn_epochs to the
        # phase-1 sweep winner (pxr_chemprop_chemeleon_phase1.py); 10 is a placeholder.
        hyperparameters={"uq_version": "v1", "from_foundation": "CheMeleon", "freeze_mpnn_epochs": 10},
    )
    m.set_owner("open_admet_pxr")
    end = m.to_endpoint(tags=tags)
    end.set_owner("open_admet_pxr")
    end.test_inference()
    end.cross_fold_inference()

# Predict the blinded phase-2 test set -> submission CSV (SMILES, Molecule Name, pEC50)
blinded = PublicData().get("comp_chem/openadmet_pxr/pxr_test_blinded")[["molecule_name", "smiles"]]
preds = Endpoint(model_name).inference(blinded)
submission = preds[["smiles", "molecule_name", "prediction"]].rename(
    columns={"smiles": "SMILES", "molecule_name": "Molecule Name", "prediction": "pEC50"}
)
submission.to_csv("phase2_chemprop_chemeleon_submission.csv", index=False)
print(f"Wrote phase2_chemprop_chemeleon_submission.csv ({len(submission)} rows)")
