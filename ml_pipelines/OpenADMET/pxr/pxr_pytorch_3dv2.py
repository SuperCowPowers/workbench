"""PXR PyTorch UQ models on 2D + curated xTB 3D-v2 features (the openadmet_pxr_f2 set).

Pulled from storage and updated for the v2 3D descriptors. Consumes the shared
`openadmet_pxr_f2` FeatureSet (train + revealed phase-1, featurized through
`smiles-to-2d-3d-v2` = RDKit/Mordred 2D + curated xTB 3D). An XGBoost UQ model is
built first to supply SHAP rankings for the reduced PyTorch variants:

    - pxr-2d-3dv2-reg-xgb           XGBoost UQ, full 2D+3D-v2 feature list (SHAP source)
    - pxr-2d-3dv2-reg-pytorch-<N>   PyTorch UQ, all N features
    - pxr-2d-3dv2-reg-pytorch-100   PyTorch UQ, top-100 non-zero SHAP

f2 already CONTAINS the phase-1 rows (split == "phase1_test"), so each model holds
them out of training via `validation_ids` (held-out never trains the model), then a
`pxr_phase1_test` capture is run on exactly those rows — pulled straight from the
FeatureSet (features already computed; no re-inference) — for honest held-out RAE.

Run after the FeatureSet exists:  python ../pxr_feature_sets.py  (builds f2)
"""

import logging

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType
from workbench.utils.chem_utils.mol_descriptors_3d_v2 import get_3d_v2_feature_names

log = logging.getLogger("workbench")

# ─── Config ───────────────────────────────────────────────────────────────────
FS_NAME = "openadmet_pxr_f2"
VARIANT = "2d-3dv2"  # used in model/endpoint names: pxr-2d-3dv2-reg-*
ENDPOINT_2D = "smiles-to-2d-v1"  # 2D feature columns; 3D-v2 columns come from get_3d_v2_feature_names()

TARGET_COL = "pec50"
SMILES_COL = "smiles"
ID_COL = "molecule_name"
SPLIT_COL = "split"
TAGS = ["openadmet_pxr", "activity", "regression", "3dv2"]
CAPTURE_NAME = "pxr_phase1_test"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def feature_list(df) -> list[str]:
    """2D columns + curated 3D-v2 columns, restricted to those present in the FeatureSet."""
    cols = list(Endpoint(ENDPOINT_2D).output_columns()) + list(get_3d_v2_feature_names())
    return [c for c in cols if c in df.columns]


def top_n_shap(model_name: str, n: int) -> list[str]:
    """Top-N non-zero SHAP features from a trained XGBoost model."""
    return [feat for feat, imp in Model(model_name).shap_importance() if imp != 0.0][:n]


def build(fs, name, framework, feats, validation_ids, extra_tags, desc) -> None:
    """Create the model + endpoint and run the standard validation captures.

    `validation_ids` holds the phase1_test rows out of training so they never train.
    """
    log.info(f"Creating {framework.name} model: {name}  ({len(feats)} features)")
    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=framework,
        target_column=TARGET_COL,
        feature_list=feats,
        description=desc,
        tags=TAGS + extra_tags,
        validation_ids=validation_ids,
        hyperparameters={"uq_version": "v1"},  # v1 = proximity-augmented RF error model
    )
    model.set_owner("open_admet_pxr")
    end = model.to_endpoint(tags=TAGS + extra_tags, max_concurrency=1)
    end.set_owner("open_admet_pxr")
    end.test_inference()
    end.cross_fold_inference()


def capture_phase1(name: str, test_df) -> None:
    """Run the held-out Analog Set 1 capture on an existing endpoint (idempotent overwrite)."""
    end = Endpoint(name)
    if not end.exists():
        log.warning(f"Endpoint '{name}' not found — skipping '{CAPTURE_NAME}' capture")
        return
    log.info(f"Capturing '{CAPTURE_NAME}' on {name} ({len(test_df)} held-out rows)")
    end.inference(test_df, capture_name=CAPTURE_NAME)


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    fs = FeatureSet(FS_NAME)
    if not fs.exists():
        raise RuntimeError(f"FeatureSet '{FS_NAME}' not found. Run pxr_feature_sets.py first.")

    df = fs.pull_dataframe()
    feats = feature_list(df)
    phase1 = df[df[SPLIT_COL] == "phase1_test"]
    validation_ids = list(phase1[ID_COL])  # held-out validation set (not trained)
    test_df = phase1[[ID_COL, SMILES_COL, TARGET_COL] + feats]  # features already present — no re-inference
    log.info(f"=== {VARIANT} — {FS_NAME}  ({len(feats)} features, {len(test_df)} held-out rows) ===")

    # XGBoost UQ — full feature list (also the SHAP source for the reduced PyTorch variants).
    xgb_name = f"pxr-{VARIANT}-reg-xgb"
    build(
        fs,
        xgb_name,
        ModelFramework.XGBOOST,
        feats,
        validation_ids,
        [VARIANT, "xgboost"],
        f"PXR pEC50 XGBoost UQ on {VARIANT} features (phase1_test held out)",
    )
    capture_phase1(xgb_name, test_df)

    # PyTorch UQ — all features, top-100 SHAP.
    for count, preset, label in [
        (len(feats), feats, f"all {len(feats)} features"),
        (100, None, "top-100 non-zero SHAP"),
    ]:
        name = f"pxr-{VARIANT}-reg-pytorch-{count}"
        selected = preset if preset is not None else top_n_shap(xgb_name, count)
        build(
            fs,
            name,
            ModelFramework.PYTORCH,
            selected,
            validation_ids,
            [VARIANT, "pytorch", f"feat{count}"],
            f"PXR pEC50 PyTorch Tabular UQ ({VARIANT}) — {label}",
        )
        capture_phase1(name, test_df)

    log.info(f"Done. {VARIANT} endpoints carry the '{CAPTURE_NAME}' capture.")
