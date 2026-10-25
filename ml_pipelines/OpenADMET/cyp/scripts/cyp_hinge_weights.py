"""Fit ensemble weights to ST-RAE's own hinge, and score them honestly.

The aggregation strategies in `EnsembleSimulator` are hand-specified -- a flat mean,
inverse-MAE, confidence-weighted. None is fitted to the metric the board reports. This fits
weights directly against the hinge: per compound, zero inside the credible interval and the
distance to the nearer edge outside it.

Weights are fitted on *deltas from an incumbent* rather than on the predictions themselves,
so a fit that learns nothing returns the incumbent instead of drifting off it. The incumbent
here is the calibrated-confidence-weighted combination, which is what the submission ships.

Two scores are printed per isoform and they mean different things:

  fitted      weights fitted and scored on the same rows. Comparable to what a competitor's
              notebook prints, and optimistic for the same reason.
  nested      outer 5-fold: weights fitted on 4/5 of the rows with L2 chosen by an inner
              5-fold, then scored on the held-out fifth. This is the number to believe.

A gap between them is the overfitting this fit is prone to; `--l2 0` widens it deliberately.

    python cyp_hinge_weights.py
    python cyp_hinge_weights.py --isoforms CYP2D6 --l2-grid 0 0.1 1.0
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from workbench.utils.ensemble_simulator import EnsembleSimulator
from workbench.utils.ensemble_utils import conf_weights_with_fallback
from workbench.utils.metrics_utils import soft_threshold_rae

from cyp_ensemble import MEMBERS, member_frame

ID_COLUMN = "molecule_name"
L2_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]


def hinge(y_pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-compound soft-threshold error: zero inside the interval, distance outside."""
    return np.maximum(0.0, lo - y_pred) + np.maximum(0.0, y_pred - hi)


def fit_weights(deltas: np.ndarray, base: np.ndarray, lo: np.ndarray, hi: np.ndarray, l2: float) -> np.ndarray:
    """Weights over member deltas minimising mean hinge plus an L2 penalty.

    Powell, matching the derivative-free choice the hinge's kinks call for. Weights are
    unconstrained in sign: the penalty is what keeps the fit from chasing the rows it sees.
    """

    def objective(w):
        return hinge(base + deltas @ w, lo, hi).mean() + l2 * float(w @ w)

    start = np.zeros(deltas.shape[1])
    result = minimize(objective, start, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 20000})
    return result.x


def pick_l2(deltas, base, lo, hi, grid, seed: int) -> float:
    """The grid value with the best mean held-out hinge over an inner 5-fold."""
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for l2 in grid:
        fold_scores = []
        for tr, va in inner.split(deltas):
            w = fit_weights(deltas[tr], base[tr], lo[tr], hi[tr], l2)
            fold_scores.append(hinge(base[va] + deltas[va] @ w, lo[va], hi[va]).mean())
        scores.append(np.mean(fold_scores))
    return grid[int(np.argmin(scores))]


def incumbent_vector(iso: str, members: list[str], frame: pd.DataFrame) -> np.ndarray:
    """The calibrated-confidence-weighted combination, as the submission builds it."""
    sim = EnsembleSimulator(members, id_column=ID_COLUMN, target=f"{iso.lower()}_pic50_direct_inhibition")
    config = sim.get_best_strategy_config(select_by="spearman")
    corr_scale = np.array([config["corr_scale"][m] for m in members])
    fallback_w = np.array([config["model_weights"][m] for m in members])
    conf = frame[[f"{m}__conf" for m in members]].to_numpy(float)
    pred = frame[[f"{m}__pred" for m in members]].to_numpy(float)
    return (pred * conf_weights_with_fallback(conf * corr_scale, fallback_w)).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isoforms", nargs="+", default=list(MEMBERS), choices=list(MEMBERS))
    parser.add_argument(
        "--l2-grid", nargs="+", type=float, default=L2_GRID, help="L2 values the inner folds choose from"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for iso in args.isoforms:
        members = MEMBERS[iso]
        frame = member_frame(iso, members)
        y = frame["label"].to_numpy(float)
        lo, hi = frame["lo"].to_numpy(float), frame["hi"].to_numpy(float)
        pred = frame[[f"{m}__pred" for m in members]].to_numpy(float)

        base = incumbent_vector(iso, members, frame)
        deltas = pred - base[:, None]

        print(f"\n=== {iso}: {len(frame):,} rows, {len(members)} members")
        print(
            f"  incumbent (calibrated conf)  ST-RAE {soft_threshold_rae(y, base, lo, hi):.4f}"
            f"  spearman {spearmanr(y, base).statistic:.4f}"
        )
        print(
            f"  flat mean                    ST-RAE {soft_threshold_rae(y, pred.mean(axis=1), lo, hi):.4f}"
            f"  spearman {spearmanr(y, pred.mean(axis=1)).statistic:.4f}"
        )

        l2 = pick_l2(deltas, base, lo, hi, args.l2_grid, args.seed)
        w_all = fit_weights(deltas, base, lo, hi, l2)
        fitted = base + deltas @ w_all
        print(
            f"  hinge fit, L2 {l2:<6g}        ST-RAE {soft_threshold_rae(y, fitted, lo, hi):.4f}"
            f"  spearman {spearmanr(y, fitted).statistic:.4f}   <- fitted on these rows"
        )

        outer = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        held = np.empty_like(base)
        for tr, va in outer.split(deltas):
            inner_l2 = pick_l2(deltas[tr], base[tr], lo[tr], hi[tr], args.l2_grid, args.seed)
            w = fit_weights(deltas[tr], base[tr], lo[tr], hi[tr], inner_l2)
            held[va] = base[va] + deltas[va] @ w
        print(
            f"  hinge fit, nested            ST-RAE {soft_threshold_rae(y, held, lo, hi):.4f}"
            f"  spearman {spearmanr(y, held).statistic:.4f}   <- the number to believe"
        )

        print("  weights on member deltas:")
        for name, w in zip(members, w_all):
            print(f"    {name.replace('cyp-reg-', ''):<34}{w:+.4f}")
        print(f"    {'sum':<34}{w_all.sum():+.4f}")


if __name__ == "__main__":
    main()
