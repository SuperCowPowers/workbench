"""Train and deploy the PXR activity-track model sweep.

Runs XGBoost + PyTorch regression on each of the three FeatureSets
(``_2d`` / ``_3d`` / ``_2d_3d``), plus a single Chemprop D-MPNN that only
consumes SMILES (so training it separately per FeatureSet would just
produce the same model three times).

Layout per FeatureSet variant:

    pxr_activity_2d         ─┬─>  pxr-2d-reg-xgb        (XGBoost UQ  — all 313 features)
                             └─>  pxr-2d-reg-pytorch    (PyTorch UQ  — top-50 non-zero SHAP from XGB)

    pxr_activity_3d         ─┬─>  pxr-3d-reg-xgb        (XGBoost UQ  — all 74 features)
                             └─>  pxr-3d-reg-pytorch    (PyTorch UQ  — top-50 non-zero SHAP)

    pxr_activity_2d_3d      ─┬─>  pxr-2d-3d-reg-xgb     (XGBoost UQ  — all 387 features)
                             └─>  pxr-2d-3d-reg-pytorch (PyTorch UQ  — top-50 non-zero SHAP)

    pxr_activity_2d_3d      ─> pxr-reg-chemprop         (Chemprop D-MPNN — SMILES only, one copy)

Each deployed endpoint runs auto / full / cross-fold inference so validation
metrics (MAE, RAE, R², Spearman, Kendall) are captured for leaderboard scoring.

Usage (with WORKBENCH_CONFIG set, e.g. scp_sandbox.json):

    python all_models.py                  # create anything missing
    python all_models.py --rebuild        # force-rebuild every model/endpoint

Prereq: ``create_feature_sets.py`` has already built the three FeatureSets
and ``smiles-to-2d-v1`` / ``smiles-to-3d-full-v1`` have registered their
feature lists via :func:`register_features`.
"""

import argparse
import logging

from workbench.api import FeatureSet, Model, ModelFramework, ModelType
from workbench.utils.feature_endpoint_utils import get_endpoint_features

log = logging.getLogger("openadmet_pxr.all_models")

# ─── Names ──────────────────────────────────────────────────────────────────
FEATURE_SETS = {
    "2d": "openadmet_pxr_activity_2d",
    "3d": "openadmet_pxr_activity_3d",
    "2d_3d": "openadmet_pxr_activity_2d_3d",
}
CHEMPROP_FS = FEATURE_SETS["2d_3d"]  # Chemprop uses smiles only; any FS works — pick the richest

TARGET_COL = "pec50"
SMILES_COL = "smiles"
TAGS_BASE = ["openadmet_pxr", "activity", "regression"]

# Feature-endpoint → feature-list key (registered by register_features)
ENDPOINT_2D = "smiles-to-2d-v1"
ENDPOINT_3D = "smiles-to-3d-full-v1"


# ─── Helpers ────────────────────────────────────────────────────────────────


def _feature_list_for_variant(variant: str) -> list[str]:
    """Assemble the feature column list for a FeatureSet variant by looking
    up the contributing feature endpoints' registered lists."""
    parts = {
        "2d": [ENDPOINT_2D],
        "3d": [ENDPOINT_3D],
        "2d_3d": [ENDPOINT_2D, ENDPOINT_3D],
    }[variant]
    features: list[str] = []
    for ep in parts:
        cols = get_endpoint_features(ep)
        if not cols:
            raise RuntimeError(
                f"No feature list registered for '{ep}' at /workbench/feature_lists/{ep}. "
                f"Run register_features on that endpoint first."
            )
        features.extend(cols)
    return features


def _top_n_shap(model_name: str, n: int = 50) -> list[str]:
    """Pull top-N non-zero SHAP features from a trained XGBoost model."""
    importances = Model(model_name).shap_importance()
    non_zero = [feat for feat, imp in importances if imp != 0.0]
    return non_zero[:n]


def _deploy_and_validate(model, variant_tag: str, framework_tag: str) -> None:
    """Create the endpoint and kick off the three validation inference runs."""
    end = model.to_endpoint(tags=TAGS_BASE + [variant_tag, framework_tag], max_concurrency=1)
    end.set_owner("open_admet_pxr")
    end.auto_inference()
    end.full_inference()
    end.cross_fold_inference()


def _create_xgb_and_pytorch(variant: str, rebuild: bool) -> None:
    """Create the XGBoost baseline and a SHAP-pruned PyTorch model for one variant."""
    fs_name = FEATURE_SETS[variant]
    fs = FeatureSet(fs_name)
    if not fs.exists():
        raise RuntimeError(f"FeatureSet '{fs_name}' not found. Run create_feature_sets.py first.")

    feature_cols = _feature_list_for_variant(variant)
    log.info(f"\n=== {variant} variant — {fs_name}  ({len(feature_cols)} features) ===")

    # 1. XGBoost UQ — uses the full feature list.
    xgb_name = f"pxr-{variant.replace('_', '-')}-reg-xgb"
    if rebuild or not Model(xgb_name).exists():
        log.info(f"Creating XGBoost model: {xgb_name}")
        xgb = fs.to_model(
            name=xgb_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column=TARGET_COL,
            feature_list=feature_cols,
            description=f"PXR pEC50 XGBoost UQ model on {variant} features",
            tags=TAGS_BASE + [variant, "xgboost"],
        )
        xgb.set_owner("open_admet_pxr")
        _deploy_and_validate(xgb, variant, "xgboost")

    # 2. PyTorch UQ — uses top-50 non-zero SHAP features from the XGB above.
    pytorch_name = f"pxr-{variant.replace('_', '-')}-reg-pytorch"
    if rebuild or not Model(pytorch_name).exists():
        top_features = _top_n_shap(xgb_name, n=50)
        log.info(f"Creating PyTorch model: {pytorch_name}  (top-{len(top_features)} SHAP features)")
        pyt = fs.to_model(
            name=pytorch_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.PYTORCH,
            target_column=TARGET_COL,
            feature_list=top_features,
            description=f"PXR pEC50 PyTorch Tabular UQ on top-50 SHAP from {xgb_name}",
            tags=TAGS_BASE + [variant, "pytorch"],
        )
        pyt.set_owner("open_admet_pxr")
        _deploy_and_validate(pyt, variant, "pytorch")


def _create_chemprop(rebuild: bool) -> None:
    """Create a single Chemprop D-MPNN — consumes SMILES only, so we train once."""
    fs = FeatureSet(CHEMPROP_FS)
    if not fs.exists():
        raise RuntimeError(f"FeatureSet '{CHEMPROP_FS}' not found. Run create_feature_sets.py first.")

    model_name = "pxr-reg-chemprop"
    if rebuild or not Model(model_name).exists():
        log.info(f"\n=== Chemprop — {model_name} (SMILES only, trained on {CHEMPROP_FS}) ===")
        cp = fs.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=TARGET_COL,
            feature_list=[SMILES_COL],
            description="PXR pEC50 Chemprop D-MPNN UQ model (SMILES only)",
            tags=TAGS_BASE + ["chemprop"],
        )
        cp.set_owner("open_admet_pxr")
        _deploy_and_validate(cp, variant_tag="smiles", framework_tag="chemprop")


# ─── Entrypoint ─────────────────────────────────────────────────────────────


def main(rebuild: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    for variant in ("2d", "3d", "2d_3d"):
        _create_xgb_and_pytorch(variant, rebuild=rebuild)

    _create_chemprop(rebuild=rebuild)

    log.info("\nDone. Models + endpoints deployed:")
    for variant in ("2d", "3d", "2d_3d"):
        v = variant.replace("_", "-")
        log.info(f"  pxr-{v}-reg-xgb")
        log.info(f"  pxr-{v}-reg-pytorch")
    log.info("  pxr-reg-chemprop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild every model and endpoint.")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
