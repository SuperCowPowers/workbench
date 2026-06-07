"""PXR Chemprop D-MPNN activity model (SMILES only).

Consumes fs:openadmet_pxr_activity_2d_3d (any FeatureSet works — Chemprop reads
SMILES directly; we pick the richest so the FeatureSet is shared with the
tabular models). Trained once, since per-FeatureSet copies would be identical.

For a model that doesn't exist yet, the endpoint is created and test / full /
cross-fold inference is run. Then — whether the endpoint is new or already
deployed — a 'pxr_phase1_test' capture is run on the held-out Analog Set 1
(pxr_test_phase1_unblinded, revealed pEC50). Chemprop is SMILES-only, so the
held-out set needs no featurization.

Self-contained for AWS Batch (the launcher uploads only this script).
"""

import logging

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData

log = logging.getLogger("workbench")

FS_NAME = "openadmet_pxr_activity_2d_3d"
MODEL_NAME = "pxr-reg-chemprop"
TARGET_COL = "pec50"
SMILES_COL = "smiles"
ID_COL = "molecule_name"
TAGS = ["openadmet_pxr", "activity", "regression"]

PHASE1_KEY = "comp_chem/openadmet_pxr/pxr_test_phase1_unblinded"
CAPTURE_NAME = "pxr_phase1_test"


def phase1_test_df():
    """Held-out Analog Set 1 — SMILES + revealed pEC50 (no featurization needed for Chemprop)."""
    df = PublicData().get(PHASE1_KEY)
    return df[[ID_COL, SMILES_COL, TARGET_COL]].dropna(subset=[TARGET_COL]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    fs = FeatureSet(FS_NAME)
    if not fs.exists():
        raise RuntimeError(f"FeatureSet '{FS_NAME}' not found. Run pxr_feature_sets.py first.")

    test_df = phase1_test_df()

    if Model(MODEL_NAME).exists():
        log.info(f"Model '{MODEL_NAME}' already exists — skipping build")
    else:
        log.info(f"Creating Chemprop model '{MODEL_NAME}' (SMILES only) on {FS_NAME} …")
        cp = fs.to_model(
            name=MODEL_NAME,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=TARGET_COL,
            feature_list=[SMILES_COL],
            description="PXR pEC50 Chemprop D-MPNN UQ model (SMILES only)",
            tags=TAGS + ["chemprop"],
        )
        cp.set_owner("open_admet_pxr")
        end = cp.to_endpoint(tags=TAGS + ["smiles", "chemprop"], max_concurrency=1)
        end.set_owner("open_admet_pxr")
        end.test_inference()
        end.full_inference()
        end.cross_fold_inference()

    # Held-out capture — runs whether the endpoint was just built or already deployed.
    end = Endpoint(MODEL_NAME)
    if end.exists():
        log.info(f"Capturing '{CAPTURE_NAME}' on {MODEL_NAME} ({len(test_df)} held-out rows)")
        end.inference(test_df, capture_name=CAPTURE_NAME)
    else:
        log.warning(f"Endpoint '{MODEL_NAME}' not found — skipping '{CAPTURE_NAME}' capture")

    log.info(f"Done. '{MODEL_NAME}' carries the '{CAPTURE_NAME}' capture.")
