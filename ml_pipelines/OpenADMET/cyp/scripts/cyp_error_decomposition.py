"""Where does an isoform's ST-RAE loss actually come from?

ST-RAE is a ratio of two sums over compounds, so it decomposes exactly: each compound
contributes `soft_threshold_error` to the numerator and `|y - mean(y)|` to the baseline
denominator. Bucketing both by true potency says which compounds we are paying for.

This matters because the scored metric is forgiving exactly where our rank correlation is
worst. Soft-threshold error is zero anywhere inside a label's credible interval, and the
low-activity end is where the intervals are widest -- so a bad ordering down there may cost
nothing. `ci hit rate` and `ci width` are the columns that settle it.

Read `loss share` against `budget share`. A band costing more of the numerator than it
holds of the denominator is where the model is actually losing.

    python cyp_error_decomposition.py
    python cyp_error_decomposition.py --models cyp-reg-chemprop-union-p30 --isoform CYP2D6
"""

import argparse

import numpy as np
import pandas as pd
from cyp_ensemble_submit import MEMBERS
from workbench.api import FeatureSet, Model
from workbench.utils.metrics_utils import soft_threshold_error

CI_SOURCE = "openadmet_cyp_f2"  # the challenge's own credible intervals
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
BANDS = [(-np.inf, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 6.0), (6.0, np.inf)]


def oof_average(models: list, target: str) -> pd.DataFrame:
    """Averaged out-of-fold predictions with truth, over the rows the models share."""
    frames = []
    for name in models:
        model = Model(name)
        runs = model.list_inference_runs()
        run = f"cv_{target}" if f"cv_{target}" in runs else "full_cross_fold"
        if run not in runs:
            continue
        df = model.get_inference_predictions(run)
        if df is None or target not in df.columns:
            continue
        frames.append(df[["molecule_name", target, "prediction"]].dropna().set_index("molecule_name"))
    if not frames:
        raise ValueError(f"no out-of-fold capture for {target}")
    rows = sorted(set.intersection(*[set(f.index) for f in frames]))
    return pd.DataFrame(
        {
            "y": frames[0].loc[rows, target],
            "pred": np.column_stack([f.loc[rows, "prediction"] for f in frames]).mean(axis=1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="comma-separated; defaults to each isoform's ensemble pool")
    parser.add_argument("--isoform", choices=ISOFORMS, help="one isoform; defaults to all four")
    args = parser.parse_args()

    ci = FeatureSet(CI_SOURCE).pull_dataframe().set_index("molecule_name")

    for iso in [args.isoform] if args.isoform else ISOFORMS:
        target = f"{iso.lower()}_pic50_direct_inhibition"
        models = [m.strip() for m in args.models.split(",")] if args.models else MEMBERS[iso]

        df = oof_average(models, target)
        df["lo"] = ci.loc[df.index, f"{target}_ci_lower"]
        df["hi"] = ci.loc[df.index, f"{target}_ci_upper"]
        df = df.dropna()

        df["err"] = soft_threshold_error(df["pred"].values, df["lo"].values, df["hi"].values)
        df["budget"] = np.abs(df["y"] - df["y"].mean())
        numerator, denominator = df["err"].sum(), df["budget"].sum()

        print(f"\n=== {iso} ({len(df):,} rows, {len(models)} member(s)) ST-RAE {numerator / denominator:.4f} ===")
        header = f"{'band':>12}{'n':>7}{'loss share':>12}{'budget share':>14}"
        print(header + f"{'band ST-RAE':>13}{'ci hit':>9}{'ci width':>10}")
        for lo, hi in BANDS:
            band = df[(df["y"] >= lo) & (df["y"] < hi)]
            if band.empty:
                continue
            label = f"{'' if np.isinf(lo) else lo}-{'' if np.isinf(hi) else hi}"
            print(
                f"{label:>12}{len(band):>7}{band['err'].sum() / numerator:>11.1%}"
                f"{band['budget'].sum() / denominator:>14.1%}"
                f"{band['err'].sum() / band['budget'].sum():>13.3f}"
                f"{(band['err'] == 0).mean():>9.1%}"
                f"{(band['hi'] - band['lo']).mean():>10.2f}"
            )


if __name__ == "__main__":
    main()
