"""PXR phase-2 submission: multi-task Chemprop (pEC50 + logD + logP).

The weekend's best held-out model. Same multi-task recipe as the phase-1 winner
(pxr-reg-chemprop-mt-both: pEC50 primary, logD + logP auxiliaries supervising the
shared MPNN, task_weights [1.0, 0.2, 0.3]) but trains on ALL pEC50 rows (train +
revealed phase-1, nothing held out) plus the public logP/logD aux data, then
predicts the 513-compound blinded phase-2 test set and writes the submission CSV.

The `prediction` column aliases the primary (pEC50) head.
Build the FeatureSet first: python ../pxr_chemprop_mt_feature_sets.py
"""

from workbench.api import FeatureSet, ModelType, ModelFramework, PublicData

fs_name = "openadmet_pxr_mt"
model_name = "pxr-reg-chemprop-mt-phase2"
targets = ["pec50", "logp", "logd"]  # pec50 first (primary)
tags = ["openadmet_pxr", "chemprop", "multi_task", "phase2"]

m = FeatureSet(fs_name).to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=targets,
    description="PXR phase-2 multi-task Chemprop (pEC50 + logD + logP aux; trains on train + phase-1)",
    tags=tags,
    hyperparameters={"uq_version": "v1", "task_weights": [1.0, 0.2, 0.3]},
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Predict the blinded phase-2 test set -> submission CSV (SMILES, Molecule Name, pEC50)
blinded = PublicData().get("comp_chem/openadmet_pxr/pxr_test_blinded")[["molecule_name", "smiles"]]
preds = end.inference(blinded)
submission = preds[["smiles", "molecule_name", "prediction"]].rename(
    columns={"smiles": "SMILES", "molecule_name": "Molecule Name", "prediction": "pEC50"}
)
submission.to_csv("phase2_chemprop_mt_submission.csv", index=False)
print(f"Wrote phase2_chemprop_mt_submission.csv ({len(submission)} rows)")
