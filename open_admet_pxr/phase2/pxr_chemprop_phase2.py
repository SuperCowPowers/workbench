"""PXR phase-2 model: Chemprop on the shared FeatureSet (openadmet_pxr_f1).

Trains on ALL rows (train + revealed phase-1 — no sample weights), predicts the
513-compound blinded phase-2 test set, and writes a submission CSV.
Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Endpoint, ModelType, ModelFramework, PublicData

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-phase2"
tags = ["openadmet_pxr", "chemprop", "phase2"]

m = FeatureSet(fs_name).to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column="pec50",
    description="PXR phase-2 pEC50 Chemprop (SMILES only; trains on all rows = train + phase-1)",
    tags=tags,
    hyperparameters={"uq_version": "v1"},  # active confidence = v1; v0/v2 also saved
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
submission.to_csv("phase2_chemprop_submission.csv", index=False)
print(f"Wrote phase2_chemprop_submission.csv ({len(submission)} rows)")
