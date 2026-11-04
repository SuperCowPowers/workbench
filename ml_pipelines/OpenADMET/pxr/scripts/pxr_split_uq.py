"""Scaffold vs Butina fold splits: their effect on UQ, out of fold and on the phase-1 holdout.

The UQ calibrator (v1: an error model plus normalized conformal scale factors) is fit on
the out-of-fold residuals, so the fold split decides what "typical error" it learns. This
scores each framework's scaffold and Butina builds (`--split` on the phase-1 scripts) on
both populations:

  oof   full_cross_fold, the rows the calibrator was fit on. Coverage there is nominal by
        construction, so only accuracy and ranking are read.
  test  pxr_phase1_test, 253 compounds no fold trained on. Captures are column-trimmed, so
        intervals come from the model's own UQ applied to the captured prediction and std.

Columns:
  rae / mae      accuracy
  rho_conf       spearman(confidence, |error|); more negative = confidence orders error better
  rho_std        spearman(prediction_std, |error|); the raw ensemble signal under the calibrator
  err_ratio      MAE of the least-confident half over the most-confident half
  cov50..cov95   fraction of rows inside the nominal interval
  width90        median width of the 90% interval
  scale90        the calibrator's 90% conformal scale factor

Deltas are Butina minus scaffold on the same 253 test compounds, paired bootstrap 95% CI.

Usage:
    python pxr_split_uq.py
    python pxr_split_uq.py --frameworks chemprop
"""

import argparse
from functools import cache

import pandas as pd
from scipy.stats import spearmanr
from workbench.api import FeatureSet, Model
from workbench.endpoints.uq_regression import uq_query_df
from workbench.utils.metrics_utils import bootstrap_compare

FRAMEWORKS = {
    "chemprop": "pxr-reg-chemprop-phase1",
    "tabicl": "pxr-reg-tabicl-monroe-readout-phase1",
}
SPLITS = ["scaffold", "butina"]
ID, TARGET = "molecule_name", "pec50"
INTERVALS = {50: ("q_25", "q_75"), 80: ("q_10", "q_90"), 90: ("q_05", "q_95"), 95: ("q_025", "q_975")}


def rae(f: pd.DataFrame) -> float:
    y = f[TARGET]
    return float((f["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(f: pd.DataFrame) -> float:
    return float((f["prediction"] - f[TARGET]).abs().mean())


def rho_conf(f: pd.DataFrame) -> float:
    return float(spearmanr(f["confidence"], (f["prediction"] - f[TARGET]).abs()).statistic)


def rho_std(f: pd.DataFrame) -> float:
    return float(spearmanr(f["prediction_std"], (f["prediction"] - f[TARGET]).abs()).statistic)


def err_ratio(f: pd.DataFrame) -> float:
    ranked = f.sort_values("confidence")
    half = len(ranked) // 2
    return mae(ranked.iloc[:half]) / mae(ranked.iloc[-half:])


def coverage(f: pd.DataFrame, level: int) -> float:
    lo, hi = INTERVALS[level]
    return float(((f[TARGET] >= f[lo]) & (f[TARGET] <= f[hi])).mean())


def width90(f: pd.DataFrame) -> float:
    lo, hi = INTERVALS[90]
    return float((f[hi] - f[lo]).median())


@cache
def smiles_by_id(fs_name: str) -> pd.Series:
    return FeatureSet(fs_name).pull_dataframe().set_index(ID)["smiles"]


def score(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """A model's OOF frame, its test frame with UQ intervals, and its 90% scale factor."""
    model = Model(model_name)
    oof = model.get_inference_predictions("full_cross_fold").set_index(ID)
    test = model.get_inference_predictions("pxr_phase1_test").set_index(ID)
    leaked = test.index.isin(oof.index).sum()
    if leaked:
        raise ValueError(f"{model_name}: {leaked} test rows are in full_cross_fold; its UQ was calibrated on them")

    uq = model.uq_model(version="v1")
    frame = test[[TARGET, "prediction", "prediction_std"]].assign(smiles=smiles_by_id(model.get_input()))
    query = uq_query_df(uq, frame)
    uq_out = uq.predict(query, frame["prediction"].to_numpy(), frame["prediction_std"].to_numpy())
    test = frame.drop(columns="smiles").join(uq_out.set_axis(frame.index))
    return oof, test, uq.scale_factors[uq.primary_target]["0.90"]


def summarize(oof: pd.DataFrame, test: pd.DataFrame, scale90: float) -> list[dict]:
    rows = []
    for population, f in (("oof", oof), ("test", test)):
        row = {
            "population": population,
            "n": len(f),
            "rae": rae(f),
            "mae": mae(f),
            "rho_conf": rho_conf(f),
            "rho_std": rho_std(f),
            "err_ratio": err_ratio(f),
        }
        if population == "test":
            row |= {f"cov{level}": coverage(f, level) for level in INTERVALS}
            row |= {"width90": width90(f), "scale90": scale90}
        rows.append(row)
    return rows


DELTAS = [
    ("rae", rae, True),
    ("|cov90 - 0.90|", lambda f: abs(coverage(f, 90) - 0.90), True),
    ("width90", width90, True),
    ("rho_conf", rho_conf, True),
    ("err_ratio", err_ratio, False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frameworks", nargs="+", choices=list(FRAMEWORKS), default=list(FRAMEWORKS))
    args = parser.parse_args()

    for framework in args.frameworks:
        scored = {split: score(f"{FRAMEWORKS[framework]}-{split}") for split in SPLITS}
        rows = [{"split": split, **row} for split, s in scored.items() for row in summarize(*s)]
        print(f"\n=== {framework}")
        print(pd.DataFrame(rows).fillna("").to_string(index=False, float_format="{:.3f}".format))

        print("\nbutina minus scaffold, test (paired bootstrap 95% CI; p = fraction of resamples butina wins)")
        butina, scaffold = scored["butina"][1], scored["scaffold"][1]
        for name, fn, lower_is_better in DELTAS:
            cmp = bootstrap_compare(butina, scaffold, fn, lower_is_better=lower_is_better)
            ci = f"[{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]"
            print(f"  {name:<16}{cmp['delta']:+.3f} {ci}  p={cmp['p_a_better']:.2f}")


if __name__ == "__main__":
    main()
