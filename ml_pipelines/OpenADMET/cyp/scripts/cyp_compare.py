"""Compare CYP models on the analog holdout.

Model scripts run on AWS Batch and nobody reads the logs, so scoring belongs in a tool a
human invokes rather than a print at the end of a training run. This is that tool.

Reports Pearson, Spearman, prediction spread and R2 alongside ST-RAE. Pearson is the one
to read for a modelling change: it sets the ceiling on R2 (`R2 <= pearsonr^2`), so a model
that improves Pearson has improved something no amount of recalibration could reach.
ST-RAE understates a CYP2D6 gain because the holdout is built from fitted-curve compounds
and lacks the inactives where better ordering there pays.

Credible intervals are label metadata rather than features, so `output_columns()` trims
them and they never reach a capture -- they are joined back from the FeatureSet by id.

Usage:
    python cyp_compare.py                             # baseline vs every other CYP model
    python cyp_compare.py MODEL [MODEL ...]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from workbench.api import FeatureSet, Meta, Model
from workbench.utils.metrics_utils import soft_threshold_rae

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
CAPTURE = "cyp_analog_holdout"


def score(model_name: str) -> pd.DataFrame:
    """Per-isoform holdout metrics for one model."""
    model = Model(model_name)
    runs = model.list_inference_runs()
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column
    ci_cols = [c for t in TARGETS for c in (f"{t}_ci_lower", f"{t}_ci_upper") if c in fs.columns]
    labels = fs.pull_dataframe()[[id_column] + ci_cols] if ci_cols else None

    rows = []
    for target in TARGETS:
        run = f"{CAPTURE}_{target}" if f"{CAPTURE}_{target}" in runs else CAPTURE
        if run not in runs:
            continue
        df = model.get_inference_predictions(run)
        pred_col = "prediction" if "prediction" in df.columns else f"{target}_pred"
        if any(c not in df.columns for c in (target, pred_col)):
            continue
        # A multi-target model's bare capture scores every isoform off target[0].
        if run == CAPTURE and len(TARGETS) > 1 and target != TARGETS[0]:
            continue

        lower, upper = f"{target}_ci_lower", f"{target}_ci_upper"
        if labels is not None and lower in labels.columns and id_column in df.columns:
            df = df.drop(columns=[c for c in df.columns if c.endswith(("_ci_lower", "_ci_upper"))])
            df = df.merge(labels, on=id_column, how="left")
        d = df[[target, pred_col]].dropna()
        y, p = d[target].to_numpy(), d[pred_col].to_numpy()
        if len(y) < 3:
            continue

        st = np.nan
        if lower in df.columns:
            c = df[[target, pred_col, lower, upper]].dropna()
            if len(c):
                st = soft_threshold_rae(c[target], c[pred_col], c[lower], c[upper])
        rows.append(
            {
                "isoform": target.split("_")[0],
                "n": len(y),
                "pearson": pearsonr(y, p).statistic,
                "spearman": spearmanr(y, p).statistic,
                "sd_pred": p.std(),
                "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                "st_rae": st,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Model names (default: every CYP regression model)")
    args = parser.parse_args()

    names = args.models
    if not names:
        tagged = Meta().models()
        names = sorted(n for n in tagged["Model Group"] if n.startswith("cyp-reg-"))

    scored = {}
    for name in names:
        try:
            df = score(name)
        except Exception as exc:
            print(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not df.empty:
            scored[name] = df

    for metric in ("pearson", "spearman", "sd_pred", "st_rae"):
        table = pd.DataFrame({n: d.set_index("isoform")[metric] for n, d in scored.items()})
        if metric == "st_rae":
            table.loc["MACRO"] = table.mean()
        print(f"\n=== {metric} ===")
        print(table.round(3).to_string())
