"""Do predicted log2FC readouts help a TabICL model over the Monroe embedding?

The single-concentration screen is the CYP analogue of the primary screen PXR took its
readouts from: same lab, same compounds, 4,375 labels per isoform against 1,285-2,335 scored
ones. A readout is that screen's *prediction* carried as a feature, which is a different
mechanism from the auxiliary heads already measured null here -- a head keeps its own scale,
so low-range information never crosses encoder -> head -> head, while a feature column goes
straight in.

`cyp-reg-chemprop-log2fc` is screen-only, so its readouts are an independent view rather than
a distillation of a pIC50 model. Its folds hold out whole rows, so a compound's readout comes
from a model that never saw that compound at all.

`--readout-model cyp-reg-chemprop-mt-aux-100` reads the contaminated version instead: that
model's log2FC head correlates -0.75 with its own pIC50 output and only +0.42 with the measured
log2FC, so it distils itself rather than the screen.

The screen is disjoint from the blind set (0 of 748 skeletons), so readouts for the 750 would
come from the endpoint at submission time, as the Monroe embedding already does.

    python cyp_readout_spike.py
    python cyp_readout_spike.py --isoform cyp3a4 --top-variance 360
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from workbench.api import FeatureSet, Model
from workbench.endpoints.inference import decompress_features
from workbench.training.splits import get_split_indices
from workbench.training.tabicl_core import fit_tabicl

FS_NAME = "openadmet_cyp_monroe_f1"
READOUT_MODEL = "cyp-reg-chemprop-log2fc"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
RESOLUTION = {"cyp1a2": 0.043, "cyp2c9": 0.031, "cyp2d6": 0.056, "cyp3a4": 0.018}
BANDS = (4.0, 4.5)
BASE = {"n_estimators": 8, "batch_size": 1, "seed": 42, "pca_components": None}


def readout_frame(model_name: str) -> pd.DataFrame:
    """Out-of-fold log2FC predictions, one column per isoform, indexed by compound."""
    model = Model(model_name)
    out = None
    for iso in ISOFORMS:
        d = model.get_inference_predictions(f"cv_{iso}_log2fc")
        col = d[["molecule_name", "prediction"]].rename(columns={"prediction": f"{iso}_log2fc_readout"})
        out = col if out is None else out.merge(col, on="molecule_name", how="outer")
    return out.set_index("molecule_name")


def out_of_fold(X: pd.DataFrame, y: np.ndarray, folds, expanded: list, top: int) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    cfg = {**BASE, "top_variance_features": top}
    for train_idx, test_idx in folds:
        model, reducer = fit_tabicl(cfg, X.iloc[train_idx], y[train_idx], cache=False, expanded_columns=expanded)
        Xte = X.iloc[test_idx]
        pred[test_idx] = model.predict(reducer.transform(Xte) if reducer is not None else Xte)
    return pred


def paired_delta(y, a, b, n_boot=500):
    rng = np.random.default_rng(0)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        d[i] = spearmanr(y[idx], b[idx]).statistic - spearmanr(y[idx], a[idx]).statistic
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="cyp2d6", choices=ISOFORMS)
    parser.add_argument("--top-variance", type=int, default=360)
    parser.add_argument("--readout-model", default=READOUT_MODEL, help="Model supplying the log2FC readouts")
    args = parser.parse_args()

    target = f"{args.isoform}_pic50_direct_inhibition"
    df = FeatureSet(FS_NAME).pull_dataframe()
    df = df[df[target].notna()].reset_index(drop=True)

    reads = readout_frame(args.readout_model)
    read_cols = list(reads.columns)
    for c in read_cols:
        df[c] = df["molecule_name"].map(reads[c])
    covered = df[read_cols].notna().all(axis=1)
    print(f"{args.isoform}: {len(df):,} labelled rows, {int(covered.sum()):,} with all four readouts")
    df = df[covered].reset_index(drop=True)

    expanded_df, features = decompress_features(df, ["monroe"], ["monroe"])
    expanded = [c for c in features if c != "monroe"]
    y = df[target].to_numpy()
    folds = get_split_indices(df, n_splits=5, strategy="scaffold", target_column=None)
    band = np.digitize(y, BANDS)
    lo, hi = band == 0, band == 2

    arms = {
        "monroe": expanded_df[expanded],
        "monroe + readouts": pd.concat([expanded_df[expanded], df[read_cols]], axis=1),
        "readouts only": df[read_cols],
    }
    preds = {}
    for name, X in arms.items():
        t0 = time.time()
        top = args.top_variance if name != "readouts only" else None
        cfg_expanded = expanded if top else []
        preds[name] = out_of_fold(X, y, folds, cfg_expanded, top) if top else out_of_fold(X, y, folds, [], None)
        rho = spearmanr(y, preds[name]).statistic
        auc = roc_auc_score(hi[lo | hi].astype(int), preds[name][lo | hi])
        print(f"  {name:<20} rho {rho:.4f}  AUC {auc:.3f}  ({(time.time() - t0) / 60:.1f} min)")

    print(f"\nresolvable Spearman difference on {args.isoform}: {RESOLUTION[args.isoform]}")
    print("paired Spearman delta, second minus first (95% CI over compounds):")
    for a, b in (("monroe", "monroe + readouts"), ("readouts only", "monroe + readouts")):
        d, lo_ci, hi_ci = paired_delta(y, preds[a], preds[b])
        star = "" if lo_ci <= 0 <= hi_ci else "  *"
        print(f"  {a:<20} -> {b:<20} {d:+.4f}  [{lo_ci:+.4f}, {hi_ci:+.4f}]{star}")
