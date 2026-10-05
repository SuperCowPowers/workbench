"""How many Monroe columns should TabICL get, and selected how?

The TabICL members carry `top_variance_features=100`, inherited from PXR without being
checked here. On PXR, 100 / 360 / 720 scored within the noise of 253 compounds; CYP2D6 trains
on a third of the rows and CYP3A4 on half, so the answer need not carry over.

CYP3A4 is the isoform to settle it on. Its resolvable difference is 0.018, the tightest of the
four, and Monroe + TabPFN beats Monroe + TabICL there by 0.022 -- a resolved gap on the same
embedding, so the head or its feature reduction is leaving something behind.

Arms, all over the same scaffold folds so every comparison is paired:

    top<N>      the N highest-variance Monroe columns, which is what the template does
    pca<N>      StandardScaler + PCA to N components, the template's other reduction
    pca<N>_raw  PCA without the scaler, so high-variance dimensions keep their weight.
                Not reachable through hyperparameters; it is here because PXR found it
                beat plain PCA.

Runs locally through `fit_tabicl`, the same call the template makes, so an arm that wins here
transfers by setting a hyperparameter. TabICL on CPU is slow -- roughly a minute per fold per
arm at these row counts -- so `--arms` narrows the sweep.

    python cyp_monroe_ablation.py
    python cyp_monroe_ablation.py --isoform cyp2d6 --arms top100 pca100
"""

import argparse
import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from workbench.api import FeatureSet
from workbench.endpoints.inference import decompress_features
from workbench.training.splits import get_split_indices
from workbench.training.tabicl_core import fit_tabicl
from workbench.utils.metrics_utils import soft_threshold_rae

FS_NAME = "openadmet_cyp_monroe_f1"
ID_COLUMN = "molecule_name"
BANDS = (4.0, 4.5)
# Smallest out-of-fold Spearman difference each isoform can resolve (scripts/cyp_ruler_power.py).
RESOLUTION = {"cyp1a2": 0.043, "cyp2c9": 0.031, "cyp2d6": 0.056, "cyp3a4": 0.018}
ARMS = {
    "top100": {"top_variance_features": 100},
    "top200": {"top_variance_features": 200},
    "top360": {"top_variance_features": 360},
    "all720": {},
    "pca100": {"pca_components": 100},
    "pca100_raw": {"pca_components": 100, "_no_scaler": True},
}
BASE = {"n_estimators": 8, "batch_size": 1, "seed": 42, "pca_components": None, "top_variance_features": None}


def fit_raw_pca(X: pd.DataFrame, y, n: int, seed: int):
    """PCA with no StandardScaler in front, which `fit_tabicl` does not expose."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from tabicl import TabICLRegressor

    from workbench.training.foundation_models import resolve_foundation_checkpoint
    from workbench.training.tabicl_core import tabicl_device

    reducer = make_pipeline(SimpleImputer(strategy="mean"), PCA(n_components=n, random_state=seed)).fit(X)
    model = TabICLRegressor(
        n_estimators=BASE["n_estimators"],
        batch_size=BASE["batch_size"],
        kv_cache=False,
        random_state=seed,
        device=tabicl_device(),
        model_path=str(resolve_foundation_checkpoint("tabicl")),
        allow_auto_download=False,
    ).fit(reducer.transform(X), y)
    return model, reducer


def out_of_fold(arm: str, X: pd.DataFrame, y: np.ndarray, folds, expanded: list, seed: int) -> np.ndarray:
    """Out-of-fold predictions for one arm, over a fold partition shared by every arm."""
    pred = np.full(len(y), np.nan)
    cfg = {**BASE, "seed": seed, **{k: v for k, v in ARMS[arm].items() if not k.startswith("_")}}
    for train_idx, test_idx in folds:
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        if ARMS[arm].get("_no_scaler"):
            model, reducer = fit_raw_pca(Xtr, y[train_idx], cfg["pca_components"], seed)
        else:
            model, reducer = fit_tabicl(cfg, Xtr, y[train_idx], cache=False, expanded_columns=expanded)
        pred[test_idx] = model.predict(reducer.transform(Xte) if reducer is not None else Xte)
    return pred


def score(y: np.ndarray, p: np.ndarray, ci: pd.DataFrame) -> dict:
    """Ordering, band separation, and the challenge's own metric."""
    band = np.digitize(y, BANDS)
    lo, hi = band == 0, band == 2
    return {
        "spearman": spearmanr(y, p).statistic,
        "auc": roc_auc_score(hi[lo | hi].astype(int), p[lo | hi]),
        "st_rae": soft_threshold_rae(y, p, ci.iloc[:, 0], ci.iloc[:, 1]),
    }


def paired_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 500) -> tuple:
    """Spearman(b) - Spearman(a) with a 95% interval, resampling compounds in common."""
    rng = np.random.default_rng(0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = spearmanr(y[idx], b[idx]).statistic - spearmanr(y[idx], a[idx]).statistic
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="cyp3a4", choices=sorted(RESOLUTION), help="Isoform to ablate")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS), help="Arms to run")
    parser.add_argument("--seed", type=int, default=42, help="TabICL seed and fold seed")
    args = parser.parse_args()

    target = f"{args.isoform}_pic50_direct_inhibition"
    ci_cols = [f"{target}_ci_lower", f"{target}_ci_upper"]
    df = FeatureSet(FS_NAME).pull_dataframe()
    df = df[df[target].notna()].reset_index(drop=True)

    expanded_df, features = decompress_features(df, ["monroe"], ["monroe"])
    expanded = [c for c in features if c != "monroe"]
    X, y = expanded_df[expanded], df[target].to_numpy()
    folds = get_split_indices(df, n_splits=5, strategy="scaffold", target_column=None)
    print(f"{args.isoform}: {len(df):,} labelled rows, {len(expanded)} Monroe columns, 5 scaffold folds")

    preds, rows = {}, []
    for arm in args.arms:
        t0 = time.time()
        preds[arm] = out_of_fold(arm, X, y, folds, expanded, args.seed)
        rows.append({"arm": arm, **score(y, preds[arm], df[ci_cols]), "minutes": (time.time() - t0) / 60})
        print(f"  {arm:<12} {rows[-1]['spearman']:.4f}  ({rows[-1]['minutes']:.1f} min)")

    print("\n" + pd.DataFrame(rows).set_index("arm").round(4).to_string())
    print(f"\nresolvable Spearman difference on {args.isoform}: {RESOLUTION[args.isoform]}")
    print("\npaired Spearman delta, second minus first (95% CI over compounds):")
    for a, b in itertools.combinations(args.arms, 2):
        d, lo, hi = paired_delta(y, preds[a], preds[b])
        star = "" if lo <= 0 <= hi else "  *"
        print(f"  {a:<12} -> {b:<12} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")
