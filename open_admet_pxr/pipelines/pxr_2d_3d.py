"""PXR 2D+3D activity models — XGBoost UQ + PyTorch UQ on the combined descriptor set.

Consumes fs:openadmet_pxr_activity_2d_3d (built by pxr_feature_sets.py): RDKit/Mordred
2D concatenated with Boltzmann-ensemble 3D features.

Models / Endpoints:
    - pxr-2d-3d-reg-xgb           XGBoost UQ, full 2D+3D feature list
    - pxr-2d-3d-reg-pytorch-<N>   PyTorch UQ, all N features
    - pxr-2d-3d-reg-pytorch-100   PyTorch UQ, top-100 non-zero SHAP (from the XGB model)
    - pxr-2d-3d-reg-pytorch-50    PyTorch UQ, top-50 non-zero SHAP

For a model that doesn't exist yet, the endpoint is created and test / full /
cross-fold inference is run. Then — whether the endpoint is new or already
deployed — a 'pxr_phase1_test' capture is run on the held-out Analog Set 1
(pxr_test_phase1_unblinded, revealed pEC50). The held-out 3D features are
computed through an InferenceCache so they are not recomputed across runs.

Self-contained for AWS Batch (the launcher uploads only this script).
"""

import logging

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.api.inference_cache import InferenceCache

log = logging.getLogger("workbench")

# ─── Config ───────────────────────────────────────────────────────────────────
FS_NAME = "openadmet_pxr_activity_2d_3d"
VARIANT = "2d-3d"  # used in model/endpoint names: pxr-2d-3d-reg-*
ENDPOINT_2D = "smiles-to-2d-v1"
ENDPOINT_3D = "smiles-to-3d-full-v1"

TARGET_COL = "pec50"
SMILES_COL = "smiles"
ID_COL = "molecule_name"
TAGS = ["openadmet_pxr", "activity", "regression"]

PHASE1_KEY = "comp_chem/openadmet_pxr/pxr_test_phase1_unblinded"
CAPTURE_NAME = "pxr_phase1_test"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def feature_list() -> list[str]:
    """Full feature list = 2D columns concatenated with 3D columns."""
    return list(Endpoint(ENDPOINT_2D).output_columns()) + list(Endpoint(ENDPOINT_3D).output_columns())


def phase1_test_df():
    """Held-out Analog Set 1, featurized through the 2D endpoint then the cached 3D endpoint."""
    df = PublicData().get(PHASE1_KEY)
    df = df[[ID_COL, SMILES_COL, TARGET_COL]].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df = Endpoint(ENDPOINT_2D).inference(df)
    cached_3d = InferenceCache(Endpoint(ENDPOINT_3D), cache_key_column=SMILES_COL)
    return cached_3d.inference(df)


def top_n_shap(model_name: str, n: int) -> list[str]:
    """Top-N non-zero SHAP features from a trained XGBoost model."""
    return [feat for feat, imp in Model(model_name).shap_importance() if imp != 0.0][:n]


def build_if_missing(fs, name: str, framework, feats: list[str], extra_tags: list[str], desc: str) -> None:
    """Create the model + endpoint and run the standard validation captures, if missing."""
    if Model(name).exists():
        log.info(f"Model '{name}' already exists — skipping build")
        return
    log.info(f"Creating {framework.name} model: {name}  ({len(feats)} features)")
    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=framework,
        target_column=TARGET_COL,
        feature_list=feats,
        description=desc,
        tags=TAGS + extra_tags,
    )
    model.set_owner("open_admet_pxr")
    end = model.to_endpoint(tags=TAGS + extra_tags, max_concurrency=1)
    end.set_owner("open_admet_pxr")
    end.test_inference()
    end.full_inference()
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

    feats = feature_list()
    test_df = phase1_test_df()
    log.info(f"=== {VARIANT} — {FS_NAME}  ({len(feats)} features, {len(test_df)} held-out rows) ===")

    # XGBoost UQ — full feature list.
    xgb_name = f"pxr-{VARIANT}-reg-xgb"
    build_if_missing(
        fs,
        xgb_name,
        ModelFramework.XGBOOST,
        feats,
        [VARIANT, "xgboost"],
        f"PXR pEC50 XGBoost UQ model on {VARIANT} features",
    )
    capture_phase1(xgb_name, test_df)

    # PyTorch UQ — all features, top-100 SHAP, top-50 SHAP.
    for count, preset, label in [
        (len(feats), feats, f"all {len(feats)} features"),
        (100, None, "top-100 non-zero SHAP"),
        (50, None, "top-50 non-zero SHAP"),
    ]:
        name = f"pxr-{VARIANT}-reg-pytorch-{count}"
        selected = preset if preset is not None else top_n_shap(xgb_name, count)
        build_if_missing(
            fs,
            name,
            ModelFramework.PYTORCH,
            selected,
            [VARIANT, "pytorch", f"feat{count}"],
            f"PXR pEC50 PyTorch Tabular UQ ({VARIANT}) — {label}",
        )
        capture_phase1(name, test_df)

    log.info(f"Done. {VARIANT} endpoints carry the '{CAPTURE_NAME}' capture.")
