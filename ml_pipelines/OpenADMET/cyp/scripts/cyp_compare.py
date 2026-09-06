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

Scores whichever capture a model has: the analog holdout where one was written, otherwise
the out-of-fold predictions, which the ruler table rates the better instrument anyway. The
holdout was retired, so nothing built since carries it.

`--bands` splits each isoform by activity, which is where CYP2D6's problem lives. Out of
fold the union model ranks the low band at 0.159 against 0.383 above it, and the blind
population is centred at 3.107 -- so an overall Spearman averages a band we cannot order
with one we can. Reported alongside is `sd_pred` per band: a model that has given up
predicts near a constant, and that shows up in the spread before it shows up in the rank.

Usage:
    python cyp_compare.py                             # baseline vs every other CYP model
    python cyp_compare.py MODEL [MODEL ...]
    python cyp_compare.py MODEL [MODEL ...] --bands
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
# Activity bands. 4.0 is where the labels stop carrying spread -- between 4.0 and 4.5 the
# CYP2D6 labels have sd 0.11 against a median measurement std of 0.069, so they are very
# nearly tied and nothing can order them.
BANDS = [("<4.0", -np.inf, 4.0), ("4.0-4.5", 4.0, 4.5), (">=4.5", 4.5, np.inf)]
# Smallest OOF-resolvable Spearman difference per isoform (scripts/cyp_ruler_power.py).
RESOLUTION = {"cyp1a2": 0.043, "cyp2c9": 0.031, "cyp2d6": 0.056, "cyp3a4": 0.018}


def resolve_capture(runs: list, target: str) -> str | None:
    """The capture holding this target's scored rows, newest convention first."""
    for candidate in (f"{CAPTURE}_{target}", CAPTURE, f"cv_{target}", "full_cross_fold"):
        if candidate in runs:
            return candidate
    return None


def model_isoforms(model) -> set:
    """The isoforms a model actually predicts, by the prefix of its target column(s)."""
    target = model.target()
    names = target if isinstance(target, list) else [target]
    return {str(n).split("_")[0] for n in names}


def scored_frame(model, target: str):
    """One model's predictions for `target`, graded against the challenge label.

    A model may train on a derived column -- a pooled scored column, say -- so the label comes
    from the FeatureSet rather than from the capture. That holds every model to the same rows
    whatever it was fit on.
    """
    run = resolve_capture(model.list_inference_runs(), target)
    if run is None:
        return None, None
    df = model.get_inference_predictions(run)
    pred_col = "prediction" if "prediction" in df.columns else f"{target}_pred"
    if pred_col not in df.columns:
        return None, None
    # A bare capture holds one unnamed prediction column, which belongs to the model's own
    # target -- claiming it for the other three isoforms would score them against it.
    if pred_col == "prediction" and target.split("_")[0] not in model_isoforms(model):
        return None, None

    fs = FeatureSet(model.get_input())
    id_column = fs.id_column
    keep = [id_column, target] + [c for c in (f"{target}_ci_lower", f"{target}_ci_upper") if c in fs.columns]
    if id_column in df.columns and target in fs.columns:
        df = df.drop(columns=[c for c in keep if c != id_column and c in df.columns])
        df = df.merge(fs.pull_dataframe()[keep], on=id_column, how="left")
    if target not in df.columns:
        return None, None
    return df.dropna(subset=[target, pred_col]), pred_col


def score(model_name: str) -> pd.DataFrame:
    """Per-isoform holdout metrics for one model."""
    model = Model(model_name)
    rows = []
    for target in TARGETS:
        d, pred_col = scored_frame(model, target)
        if d is None or len(d) < 3:
            continue
        y, p = d[target].to_numpy(), d[pred_col].to_numpy()

        st = np.nan
        lower, upper = f"{target}_ci_lower", f"{target}_ci_upper"
        if lower in d.columns:
            c = d[[target, pred_col, lower, upper]].dropna()
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


def score_bands(model_name: str) -> pd.DataFrame:
    """Spearman and prediction spread per activity band, for one model."""
    model = Model(model_name)
    rows = []
    for target in TARGETS:
        d, pred_col = scored_frame(model, target)
        if d is None:
            continue
        for label, lo, hi in BANDS:
            band = d[(d[target] >= lo) & (d[target] < hi)]
            # Spearman on a handful of rows is not a measurement.
            rho = spearmanr(band[target], band[pred_col]).statistic if len(band) > 20 else np.nan
            rows.append(
                {
                    "isoform": target.split("_")[0],
                    "band": label,
                    "n": len(band),
                    "spearman": rho,
                    "sd_label": band[target].std() if len(band) > 1 else np.nan,
                    "sd_pred": band[pred_col].std() if len(band) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Model names (default: every CYP regression model)")
    parser.add_argument("--bands", action="store_true", help="Split each isoform by activity band")
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

    if args.bands:
        banded = {}
        for name in scored:
            b = score_bands(name)
            if not b.empty:
                banded[name] = b.set_index(["isoform", "band"])
        for metric in ("spearman", "sd_pred"):
            table = pd.DataFrame({n: d[metric] for n, d in banded.items()})
            print(f"\n=== {metric} by band ===")
            print(table.round(3).to_string())
        first = next(iter(banded.values()))
        print("\n=== band sizes and label spread (identical across models) ===")
        print(first[["n", "sd_label"]].round(3).to_string())
        print("\nresolvable Spearman difference per isoform (overall, OOF): " f"{RESOLUTION}")
