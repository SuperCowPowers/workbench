"""PXR phase-1 spike: fine-tune Atom-JEPA on pEC50 with the primary-screen readouts.

The fine-tuned counterpart of pxr-reg-chemprop-readout-phase1: same FeatureSet, same
features (SMILES + the two predicted log2FC readouts), phase1_test held out via
validation_ids. Trains with the Atom-JEPA framework on a GPU training job. No endpoint:
the template scores the held-out rows with its fold ensemble, and
pxr_atomjepa_scoreboard.py reads them back as the "validation" capture.

Run from anywhere (blocks while the training job runs):  python pxr_atomjepa_finetune.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType

fs_name = "openadmet_pxr_readout"
model_name = "pxr-reg-atomjepa-readout-phase1"
tags = ["openadmet_pxr", "atomjepa", "primary_screen", "phase1", "spike"]
HYPERPARAMETERS = {"n_folds": 1}

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

m = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.ATOMJEPA,
    feature_list=["smiles", "lfc_8um_readout", "lfc_33um_readout"],
    target_column="pec50",
    description="PXR phase-1 pEC50 fine-tuned Atom-JEPA + primary-screen log2FC readouts (spike, phase1_test held out)",
    tags=tags,
    validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
    hyperparameters=HYPERPARAMETERS,
)
m.set_owner("open_admet_pxr")
