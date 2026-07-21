"""PXR phase-1 CheMeleon freeze sweep — NEGATIVE result (kept for the record).

Chemprop warm-started from the CheMeleon foundation model, sweeping the freeze
length (full fine-tune vs linear-probe-then-fine-tune, Kumar et al. 2022). Held-
out phase1_test RAE: frz0 0.696, frz10 0.704, frz20 0.706 — all ~0.12 worse than
from-scratch chemprop (0.577). Even frz0 (CheMeleon's validated full fine-tune)
loses badly, so it's not a freeze-tuning problem: `from_foundation` pins the MPNN
to CheMeleon's pretrained dims, replacing our tuned depth=6/hidden_dim=700, and on
this small assay the from-scratch tuned MPNN wins. So CheMeleon is NOT used for
the phase-2 submission (that script was deleted); this file documents why.

Build the FeatureSet first: python ../pxr_feature_sets.py
"""

from workbench.api import FeatureSet, Endpoint, ModelType, ModelFramework

fs_name = "openadmet_pxr_f1"
FREEZE_EPOCHS = [0, 10, 20]  # 0 = CheMeleon's validated full fine-tune; >0 = LP-FT bet for OOD
base_tags = ["openadmet_pxr", "chemprop", "chemeleon", "phase1"]

fs = FeatureSet(fs_name)
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]
validation_ids = list(phase1["molecule_name"])  # held-out validation set (not trained)

for frz in FREEZE_EPOCHS:
    model_name = f"pxr-reg-chemprop-chemeleon-phase1-frz{frz}"
    tags = base_tags + [f"frz{frz}"]
    m = fs.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        feature_list=["smiles"],
        target_column="pec50",
        description=f"PXR phase-1 Chemprop, CheMeleon warm-start, freeze={frz} (phase1_test held out)",
        tags=tags,
        hyperparameters={"uq_version": "v1", "from_foundation": "CheMeleon", "freeze_mpnn_epochs": frz},
        validation_ids=validation_ids,
    )
    m.set_owner("open_admet_pxr")
    end = m.to_endpoint(tags=tags)
    end.set_owner("open_admet_pxr")
    end.test_inference()
    end.cross_fold_inference()
    # Held-out capture on the phase1_test rows (the model never trained on them)
    Endpoint(model_name).inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
