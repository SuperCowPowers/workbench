"""PXR Chemprop *hybrid* models — D-MPNN (SMILES) + top-SHAP 3D descriptors.

The plain Chemprop model (pxr_chemprop.py) is SMILES-only and is our strongest
held-out model. This script asks the 3D question in the strongest possible
setting: does *appending the highest-SHAP 3D conformer features* to the learned
graph representation move the needle, or is the SHAP importance of those features
(see the 2D-vs-3D blog) non-transferable noise?

Chemprop's hybrid mode (template: "SMILES + extra molecular descriptors") drives
the D-MPNN from the `smiles` column and concatenates every other column in
`feature_list` as scaled extra descriptors (`x_d`) ahead of the FFN. So a hybrid
is just `feature_list = [smiles] + <selected 3D features>`.

Feature selection is computed at runtime from the deployed XGBoost SHAP model
(pxr-2d-3d-reg-xgb) — we take the 3D features that land inside the overall
top-20 and top-50 SHAP ranks (≈6 and ≈16 features respectively). No hardcoding,
so it tracks the model the blog is built on.

Models / Endpoints:
    - pxr-chemprop-hybrid-3d-top20   SMILES + 3D features within top-20 SHAP (≈6)
    - pxr-chemprop-hybrid-3d-top50   SMILES + 3D features within top-50 SHAP (≈16)
    - pxr-chemprop-hybrid-3d-all     SMILES + all 74 3D features (control)

The 'all' control separates "SHAP-selected 3D helps" from "any 3D helps" — if
even all 74 3D features don't beat plain chemprop, the case is airtight.

Consumes fs:openadmet_pxr_activity_2d_3d (has smiles passthrough + all 3D cols).
Each endpoint gets test / full / cross_fold inference plus a 'pxr_phase1_test'
capture on the held-out Analog Set 1 (revealed pEC50), featurized through the
cached 3D endpoint — identical protocol to the other PXR model scripts, so the
held-out RAE is directly comparable to pxr-reg-chemprop.

Self-contained for AWS Batch (the launcher uploads only this script).
"""

import logging

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.api.inference_cache import InferenceCache

log = logging.getLogger("workbench")

# ─── Config ───────────────────────────────────────────────────────────────────
FS_NAME = "openadmet_pxr_activity_2d_3d"
ENDPOINT_3D = "smiles-to-3d-full-v1"
SHAP_MODEL = "pxr-2d-3d-reg-xgb"  # deployed XGB UQ model — source of the SHAP ranking

TARGET_COL = "pec50"
SMILES_COL = "smiles"
ID_COL = "molecule_name"
TAGS = ["openadmet_pxr", "activity", "regression", "chemprop", "hybrid"]

PHASE1_KEY = "comp_chem/openadmet_pxr/pxr_test_phase1_unblinded"
CAPTURE_NAME = "pxr_phase1_test"

# (endpoint suffix, SHAP top-N pool) — the 3D features inside that pool are appended.
# pool_n=None selects all 3D features (the "any 3D" control).
VARIANTS = [("top20", 20), ("top50", 50), ("all", None)]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def select_3d(pool_n: int | None) -> list[str]:
    """3D feature columns to append: top-`pool_n` SHAP (non-zero), or all 3D if None."""
    cols_3d = list(Endpoint(ENDPOINT_3D).output_columns())
    if pool_n is None:
        return cols_3d  # control: all 74 3D features
    cols_3d_set = set(cols_3d)
    ranked = Model(SHAP_MODEL).shap_importance()  # [(feature, mean_abs_shap), ...] descending
    return [feat for feat, imp in ranked[:pool_n] if feat in cols_3d_set and imp != 0.0]


def phase1_test_df():
    """Held-out Analog Set 1, featurized through the cached 3D endpoint (smiles + 3D cols)."""
    df = PublicData().get(PHASE1_KEY)
    df = df[[ID_COL, SMILES_COL, TARGET_COL]].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    cached_3d = InferenceCache(Endpoint(ENDPOINT_3D), cache_key_column=SMILES_COL)
    return cached_3d.inference(df)


def build_if_missing(fs, name: str, feats: list[str], extra_tags: list[str], desc: str) -> None:
    """Create the hybrid Chemprop model + endpoint and run standard captures, if missing."""
    if Model(name).exists():
        log.info(f"Model '{name}' already exists — skipping build")
        return
    log.info(f"Creating Chemprop hybrid model: {name}  (smiles + {len(feats) - 1} 3D features)")
    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=TARGET_COL,
        feature_list=feats,  # [smiles] + selected 3D columns
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
    if not Model(SHAP_MODEL).exists():
        raise RuntimeError(f"SHAP source model '{SHAP_MODEL}' not found. Run pxr_2d_3d.py first.")

    test_df = phase1_test_df()

    for suffix, pool_n in VARIANTS:
        sel_3d = select_3d(pool_n)
        name = f"pxr-chemprop-hybrid-3d-{suffix}"
        pool_desc = "all 3D" if pool_n is None else f"top-{pool_n} SHAP pool"
        log.info(f"=== {name}: smiles + {len(sel_3d)} 3D features ({pool_desc}) — {sel_3d}")
        build_if_missing(
            fs,
            name,
            [SMILES_COL] + sel_3d,
            [f"3d_{suffix}"],
            f"PXR pEC50 Chemprop D-MPNN hybrid — SMILES + {len(sel_3d)} top-SHAP 3D features ({suffix})",
        )
        capture_phase1(name, test_df)

    log.info(f"Done. Hybrid endpoints carry the '{CAPTURE_NAME}' capture.")
