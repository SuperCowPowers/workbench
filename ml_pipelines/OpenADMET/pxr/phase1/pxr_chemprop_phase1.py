"""PXR phase-1 model: Chemprop on the shared FeatureSet (openadmet_pxr_f1).

Holds the `phase1_test` rows out of training via validation_ids so the held-out
Analog Set 1 never trains the model, then captures predictions on exactly those
rows as 'pxr_phase1_test'. Build the FeatureSet first: python ../pxr_feature_sets.py

`--split` sets the fold split (the UQ calibrators are fit on its out-of-fold residuals):
    ml_pipeline_launcher pxr_chemprop_phase1.py -- --split butina
"""

import argparse

from workbench.api import FeatureSet, ModelType, ModelFramework

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--split",
    choices=["scaffold", "butina"],
    help="Fold split strategy; suffixes the model name with it, so the build lands beside the default model",
)
args = parser.parse_args()

fs_name = "openadmet_pxr_f1"
model_name = "pxr-reg-chemprop-phase1" + (f"-{args.split}" if args.split else "")
tags = ["openadmet_pxr", "chemprop", "phase1"] + ([args.split] if args.split else [])

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column="pec50",
    description="PXR phase-1 pEC50 Chemprop (SMILES only; phase1_test held out of training)",
    tags=tags,
    validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
    hyperparameters={"split_strategy": args.split} if args.split else None,
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()

# Held-out capture on the phase1_test rows (the model never trained on them)
end.inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
