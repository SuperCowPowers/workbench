"""Rewrite the CYP analog-holdout captures on the current code.

Fixes the per-target prediction columns (see `_remap_multi_target_columns`) by re-running
inference rather than retraining. `cross_fold_inference` covers the cv_* captures; this
covers the holdout ones.

Reports ST-RAE and the prediction spread per isoform. A multi-target model that failed the
remap scores every isoform off `prediction`, which is target[0] -- so four identical
numbers is the failure signature, and four different ones is the pass.

Run:  python cyp_recapture.py
"""

from workbench.api import Endpoint, FeatureSet, Model
from workbench.training.splits import analog_holdout_split
from workbench.utils.metrics_utils import soft_threshold_rae

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
CI_COLUMNS = [f"{t}_{bound}" for t in TARGETS for bound in ("ci_lower", "ci_upper")]

# Multi-task chemprop scores every isoform from one model; the XGBoost models are
# single-task, so each carries only its own isoform's capture.
MODELS = ["cyp-reg-chemprop-mt"] + [f"cyp-2d-3dv2-reg-xgb-{iso.removeprefix('cyp')}" for iso in ISOFORMS]

# The same holdout every CYP model was evaluated on — the split is deterministic.
df = FeatureSet("openadmet_cyp_f1").pull_dataframe()
holdout = df[analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)]
holdout_df = holdout[["molecule_name", "smiles"] + TARGETS + CI_COLUMNS]

for name in MODELS:
    print(f"\n=== {name} ===")
    end = Endpoint(name)
    end.test_inference()
    end.inference(holdout_df, capture_name="cyp_analog_holdout")

    model = Model(name)
    runs = model.list_inference_runs()
    seen = []
    for iso in ISOFORMS:
        target = f"{iso}_pic50_direct_inhibition"
        run = f"cyp_analog_holdout_{target}" if f"cyp_analog_holdout_{target}" in runs else "cyp_analog_holdout"
        if run not in runs or target not in model.get_inference_predictions(run).columns:
            continue
        preds = model.get_inference_predictions(run).dropna(subset=[target, "prediction"])
        st_rae = soft_threshold_rae(
            preds[target], preds["prediction"], preds[f"{target}_ci_lower"], preds[f"{target}_ci_upper"]
        )
        seen.append(round(float(st_rae), 6))
        print(f"  {iso}: ST-RAE {st_rae:.3f} | pred sd {preds['prediction'].std():.2f} | n {len(preds)}")
    if len(seen) > 1 and len(set(seen)) == 1:
        print("  MISMATCH: every isoform scored identically — the per-target remap did not take")
