"""Soft-threshold RAE for a stored inference capture.

ST-RAE is the challenge's metric but not a Workbench one -- `compute_regression_metrics`
reports rmse/mae/r2/pearsonr/spearmanr/spread_ratio/bias and nothing about credible
intervals. The `_ci_lower`/`_ci_upper` columns ride along in the CYP FeatureSets so the
scripts here can score against them; this is what does that scoring.

Reads each capture's own predictions rather than a stored column, so a disagreement means
the predictions changed, not that a metric went missing.
"""

import pandas as pd
from workbench.utils.metrics_utils import soft_threshold_rae


def capture_st_rae(model, targets: list, capture_name: str = "cyp_analog_holdout") -> pd.DataFrame:
    """Recompute ST-RAE per target from a capture's predictions.

    For a multi-target model the bare capture's `prediction` column is target[0], so only
    the per-target runs are read; scoring every target off that one column silently
    reports the primary target's number four times.

    Args:
        model: Workbench Model carrying the capture.
        targets: Target column names to score.
        capture_name: Capture to read.

    Returns:
        One row per scored target (target, st_rae, support), plus a macro row when more
        than one target scored. Empty if the capture carries no credible intervals.
    """
    runs = model.list_inference_runs()
    rows = []
    for target in targets:
        per_target = f"{capture_name}_{target}"
        if per_target in runs:
            run, pred_col = per_target, "prediction"
        elif len(targets) == 1 and capture_name in runs:
            run, pred_col = capture_name, "prediction"
        else:
            continue

        df = model.get_inference_predictions(run)
        if pred_col not in df.columns:
            pred_col = f"{target}_pred"
        lower, upper = f"{target}_ci_lower", f"{target}_ci_upper"
        needed = [target, pred_col, lower, upper]
        if any(c not in df.columns for c in needed):
            continue

        d = df[needed].dropna()
        if d.empty:
            continue
        rows.append(
            {
                "target": target,
                "st_rae": soft_threshold_rae(d[target], d[pred_col], d[lower], d[upper]),
                "support": len(d),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["target", "st_rae", "support"])
    scores = pd.DataFrame(rows)
    if len(scores) > 1:
        macro = {"target": "MACRO", "st_rae": scores["st_rae"].mean(), "support": int(scores["support"].sum())}
        scores = pd.concat([scores, pd.DataFrame([macro])], ignore_index=True)
    return scores
