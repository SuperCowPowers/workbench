"""PXR phase-1: TabICL's native intervals vs the UQModel V1 intervals built on them.

The tabicl template feeds only the width of TabICL's predictive distribution
(`prediction_std`) into UQModel V1, which then builds its own conformal intervals.
This scores both sets of intervals on the 253 `phase1_test` rows, so the choice of
which to publish as the `q_*` columns rests on coverage and width, not assumption.

Mirrors the template: scaffold-split fold models give out-of-fold predictions for V1's
calibration, one cached model on all `train` rows scores the held-out rows. Features are
the scoreboard's best arm (CheMeleon PCA256 + primary-screen readouts).

Result (253 rows; interval score = width + miss penalty, lower is better):

    nominal   native coverage / score   V1 coverage / score
    50%       54.5% / 1.518             56.1% / 1.499
    68%       71.1% / 1.904             70.4% / 1.894
    80%       79.4% / 2.307             80.2% / 2.320
    95%       94.1% / 3.425             92.1% / 3.833

A tie through 80%; at 95% the native intervals cover better and score better, a
difference of about five compounds. V1's error model leans on TabICL's width
(prediction_std importance 0.48 of five features), so the native signal carries through.
The template keeps V1 for the `q_*` columns: same contract as every other framework.

Runs locally (not via ml_pipeline_launcher), ~6 minutes on CPU:

    uv run python pxr_tabicl_uq_phase1.py
"""

import numpy as np
import pandas as pd
import torch
from chemprop import data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from workbench.api import FeatureSet
from workbench.endpoints.tabicl_utils import predict_with_std
from workbench.endpoints.uq_regression import fit_regression_uq, uq_query_df
from workbench.training.chemprop_core import load_foundation_weights
from workbench.training.splits import get_split_indices
from workbench.training.tabicl_core import fit_tabicl

fs_name = "openadmet_pxr_readout"
id_col, target = "molecule_name", "pec50"
READOUTS = ["lfc_8um_readout", "lfc_33um_readout"]
PCA_DIMS = 256
HYPERPARAMETERS = {"n_estimators": 8, "batch_size": 1, "seed": 42, "pca_components": None}

# Nominal coverage -> (lower, upper) quantile columns, as UQModel V1 names them
LEVELS = {0.50: ("q_25", "q_75"), 0.68: ("q_16", "q_84"), 0.80: ("q_10", "q_90"), 0.95: ("q_025", "q_975")}
ALPHAS = {
    "q_025": 0.025,
    "q_10": 0.10,
    "q_16": 0.16,
    "q_25": 0.25,
    "q_75": 0.75,
    "q_84": 0.84,
    "q_90": 0.90,
    "q_975": 0.975,
}


def chemeleon_embed(smiles: list[str], batch_size: int = 256) -> np.ndarray:
    """Frozen CheMeleon MPNN + mean aggregation -> one 2048-d vector per molecule."""
    mp, agg = load_foundation_weights("CheMeleon")
    mp.eval()
    dataset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
    loader = data.build_dataloader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        return np.vstack([agg(mp(batch.bmg), batch.bmg.batch).numpy() for batch in loader])


def interval_report(name: str, intervals: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """Coverage, mean width, and interval score (width + miss penalty; lower is better) per level."""
    rows = []
    for level, (lo_col, hi_col) in LEVELS.items():
        lo, hi = intervals[lo_col].to_numpy(), intervals[hi_col].to_numpy()
        miss = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
        rows.append(
            {
                "intervals": name,
                "nominal": level,
                "coverage": float(np.mean((y >= lo) & (y <= hi))),
                "mean_width": float(np.mean(hi - lo)),
                "interval_score": float(np.mean((hi - lo) + (2 / (1 - level)) * miss)),
            }
        )
    return rows


# Data: train rows fit and calibrate, phase1_test rows score
df = FeatureSet(fs_name).pull_dataframe()[[id_col, "smiles", target, "split"] + READOUTS]
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "phase1_test"].reset_index(drop=True)
print(f"{len(train)} train / {len(test)} phase1_test rows")

# CheMeleon embedding -> standardize -> PCA (fit on train only), plus the readouts
emb_train, emb_test = chemeleon_embed(train["smiles"].tolist()), chemeleon_embed(test["smiles"].tolist())
scaler = StandardScaler().fit(emb_train)
pca = PCA(n_components=PCA_DIMS, random_state=0).fit(scaler.transform(emb_train))
features = [f"pc_{i}" for i in range(PCA_DIMS)] + READOUTS
X_train = pd.DataFrame(pca.transform(scaler.transform(emb_train)), columns=features[:PCA_DIMS])
X_test = pd.DataFrame(pca.transform(scaler.transform(emb_test)), columns=features[:PCA_DIMS])
X_train[READOUTS], X_test[READOUTS] = train[READOUTS], test[READOUTS]

# Out-of-fold predictions and spreads from uncached fold models (V1's calibration set)
folds = get_split_indices(train, n_splits=5, strategy="scaffold", target_column=None, test_size=0.2, random_state=42)
oof_pred, oof_std = np.full(len(train), np.nan), np.full(len(train), np.nan)
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"Fold {fold_idx + 1}/{len(folds)}: context {len(train_idx)}, val {len(val_idx)}")
    fold_model, _ = fit_tabicl(HYPERPARAMETERS, X_train.iloc[train_idx], train[target].iloc[train_idx], kv_cache=False)
    oof_pred[val_idx], oof_std[val_idx] = predict_with_std(fold_model, X_train.iloc[val_idx])

# Served model on all train rows: held-out mean, spread, and native quantiles
model, _ = fit_tabicl(HYPERPARAMETERS, X_train, train[target], kv_cache="repr")
out = model.predict(X_test, output_type=["mean", "quantiles"], alphas=sorted(ALPHAS.values()))
native = pd.DataFrame(out["quantiles"], columns=sorted(ALPHAS, key=ALPHAS.get))
pred, std = np.asarray(out["mean"]), ((native["q_84"] - native["q_16"]) / 2).to_numpy()

# UQModel V1, calibrated on the out-of-fold rows exactly as the template does
prox_df = train[[id_col, target, "smiles"]].copy()
prox_df["in_model"] = True
uq_model = fit_regression_uq(
    per_target={
        target: {"ids": train[id_col].tolist(), "y_true": train[target].values, "y_pred": oof_pred, "y_std": oof_std}
    },
    prox_df=prox_df,
    id_column=id_col,
    features=features,
    active_version="v1",
)["uq_model"]
v1 = uq_model.predict(uq_query_df(uq_model, test), pred, std)

y = test[target].to_numpy()
report = pd.DataFrame(interval_report("tabicl_native", native, y) + interval_report("uq_v1", v1, y))
print(f"\nphase1_test MAE {np.abs(pred - y).mean():.3f}; intervals on {len(y)} rows (interval_score: lower is better)")
print(report.to_string(index=False, float_format="%.3f"))
