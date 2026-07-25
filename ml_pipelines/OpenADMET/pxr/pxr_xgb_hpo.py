"""PXR XGBoost with a hyperparameter search on 2D + curated xTB 3D-v2 features.

The HPO counterpart to the `pxr-2d-3dv2-reg-xgb` model in pxr_pytorch_3dv2.py — same
`openadmet_pxr_f2` FeatureSet, same feature list, same held-out rows — so the pair is a
direct A/B on whether tuning helps this data.

f2 already CONTAINS the phase-1 rows (split == "phase1_test"), so they are held out of
training via `validation_ids` and a `pxr_phase1_test` capture is run on exactly those
rows for honest held-out RAE.

The search objective is `cv_mae` on scaffold folds of the training rows (the default). The
phase1_test rows stay out of training AND out of the search, so the capture stays
comparable to every other model in the pipeline.

Run after the FeatureSet exists:  python pxr_feature_sets.py  (builds f2)
"""

import logging

from workbench.api import Endpoint, FeatureSet, ModelFramework, ModelType
from workbench.utils.chem_utils.mol_descriptors_3d_v2 import get_3d_v2_feature_names

log = logging.getLogger("workbench")

FS_NAME = "openadmet_pxr_f2"
MODEL_NAME = "pxr-2d-3dv2-reg-xgb-hpo"
ENDPOINT_2D = "smiles-to-2d-v1"  # 2D feature columns; 3D-v2 columns come from get_3d_v2_feature_names()

TARGET_COL = "pec50"
SMILES_COL = "smiles"
ID_COL = "molecule_name"
SPLIT_COL = "split"
TAGS = ["openadmet_pxr", "activity", "regression", "3dv2", "xgboost", "hpo"]
CAPTURE_NAME = "pxr_phase1_test"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    fs = FeatureSet(FS_NAME)
    if not fs.exists():
        raise RuntimeError(f"FeatureSet '{FS_NAME}' not found. Run pxr_feature_sets.py first.")

    df = fs.pull_dataframe()
    cols = list(Endpoint(ENDPOINT_2D).output_columns()) + list(get_3d_v2_feature_names())
    feats = [c for c in cols if c in df.columns]
    phase1 = df[df[SPLIT_COL] == "phase1_test"]
    test_df = phase1[[ID_COL, SMILES_COL, TARGET_COL] + feats]  # features already present — no re-inference
    log.info(f"=== {MODEL_NAME} — {FS_NAME}  ({len(feats)} features, {len(test_df)} held-out rows) ===")

    m = fs.to_model(
        name=MODEL_NAME,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column=TARGET_COL,
        feature_list=feats,
        description=f"PXR pEC50 XGBoost UQ on 2d-3dv2 features, hyperparameter-searched ({CAPTURE_NAME} held out)",
        tags=TAGS,
        validation_ids=list(phase1[ID_COL]),  # held-out validation set (not trained)
        hyperparameters={
            "uq_version": "v1",  # v1 = proximity-augmented RF error model
            "hpo": {
                # The base training image carries optuna, not ray — and one XGBoost fit
                # already spreads across every core, so the search is serial by design.
                "backend": "optuna",
                "n_trials": 100,
                # Score on scaffold folds of the training rows, never on the held-out
                # phase1_test rows — tuning on those would cost them their role as a benchmark.
                "metric": "cv_mae",
                # The search only shortlists: its top rerank_top_k configs are re-scored against
                # these hyperparameters as-is, and the winner of *that* is published. So a search
                # that finds nothing real publishes the untuned baseline.
                "rerank_top_k": 5,
            },
        },
    )
    m.set_owner("open_admet_pxr")
    end = m.to_endpoint(tags=TAGS, max_concurrency=1)
    end.set_owner("open_admet_pxr")
    end.test_inference()
    end.cross_fold_inference()

    # Held-out capture on the phase1_test rows (the model never trained on them).
    log.info(f"Capturing '{CAPTURE_NAME}' on {MODEL_NAME} ({len(test_df)} held-out rows)")
    end.inference(test_df, capture_name=CAPTURE_NAME)
