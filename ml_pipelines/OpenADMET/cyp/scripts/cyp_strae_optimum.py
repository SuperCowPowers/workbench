"""Derive each isoform's ST-RAE-optimal placement instead of sampling it on the board.

Placement is an affine transform, so a submission is fully described by the mean and sd
it is placed at. R2 fixes those analytically -- centre on the truth, spread at `rho*sd`.
ST-RAE does not, because soft-threshold error is zero inside a label's credible interval
and the intervals vary by potency, so the objective is a function of the CI structure
rather than of the moments. Board sampling is how we have located it so far, at one
submission per probe.

This locates it offline. Out-of-fold predictions come with the challenge's own intervals,
so ST-RAE can be evaluated over a grid of placements directly.

The out-of-fold rows are hit-enriched where the blind half is not, so every statistic is
importance-weighted onto the blind population (`BLIND_MOMENTS`, normal) before the search.
The weighting assumes the CI-vs-potency relationship carries over, which is the load-bearing
assumption here -- it reweights compounds we have, it cannot invent low-activity ones.

Validate before trusting it: CYP2D6's derived optimum should land near the 3.57 / 0.90 that
board probes found. `--validate` prints that comparison.

    python cyp_strae_optimum.py
    python cyp_strae_optimum.py --isoform CYP2D6 --validate
"""

import argparse

import numpy as np
from cyp_ensemble_submit import MEMBERS
from cyp_error_decomposition import CI_SOURCE, ISOFORMS, oof_average
from cyp_recalibrate import BLIND_MOMENTS, STRAE_MOMENTS
from scipy.stats import gaussian_kde, norm, pearsonr
from workbench.api import FeatureSet
from workbench.utils.metrics_utils import soft_threshold_error

MEAN_GRID = np.arange(2.0, 6.01, 0.05)
SD_GRID = np.arange(0.20, 2.51, 0.05)


def blind_weights(y: np.ndarray, iso: str) -> np.ndarray:
    """Importance weights carrying the out-of-fold labels onto the blind population."""
    moments = BLIND_MOMENTS[iso]
    target = norm.pdf(y, moments["mean"], moments["sd"])
    have = gaussian_kde(y)(y)
    w = target / np.clip(have, 1e-12, None)
    return w / w.sum()


def strae(pred: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, w: np.ndarray) -> float:
    """Weighted ST-RAE: soft-threshold error over the mean-predictor baseline."""
    numerator = (w * soft_threshold_error(pred, lo, hi)).sum()
    denominator = (w * np.abs(y - (w * y).sum())).sum()
    return numerator / denominator


def place(pred: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Affine transform onto a target mean and sd. Leaves every compound's rank alone."""
    return mean + (pred - pred.mean()) * (sd / pred.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", choices=ISOFORMS, help="one isoform; defaults to all four")
    parser.add_argument("--validate", action="store_true", help="compare against the board-probed placements")
    args = parser.parse_args()

    ci = FeatureSet(CI_SOURCE).pull_dataframe().set_index("molecule_name")

    for iso in [args.isoform] if args.isoform else ISOFORMS:
        target = f"{iso.lower()}_pic50_direct_inhibition"
        df = oof_average(MEMBERS[iso], target)
        df["lo"] = ci.loc[df.index, f"{target}_ci_lower"]
        df["hi"] = ci.loc[df.index, f"{target}_ci_upper"]
        df = df.dropna()

        y, pred, lo, hi = (df[c].values for c in ("y", "pred", "lo", "hi"))
        w = blind_weights(y, iso)
        rho = pearsonr(y, pred).statistic

        grid = np.array([[strae(place(pred, m, s), y, lo, hi, w) for s in SD_GRID] for m in MEAN_GRID])
        i, j = np.unravel_index(grid.argmin(), grid.shape)
        best_mean, best_sd = MEAN_GRID[i], SD_GRID[j]

        # R2's optimum, on the same weighted population: centre on the truth, spread at rho*sd.
        r2_mean = (w * y).sum()
        r2_sd = rho * np.sqrt((w * (y - r2_mean) ** 2).sum())

        print(f"\n=== {iso} ({len(df):,} rows, OOF pearson {rho:.3f}) ===")
        print(f"{'placement':<26}{'mean':>7}{'sd':>7}{'ST-RAE':>10}")
        print(f"{'ST-RAE optimum':<26}{best_mean:>7.2f}{best_sd:>7.2f}{grid[i, j]:>10.4f}")
        r2_score = strae(place(pred, r2_mean, r2_sd), y, lo, hi, w)
        print(f"{'R2 optimum':<26}{r2_mean:>7.2f}{r2_sd:>7.2f}{r2_score:>10.4f}")
        if args.validate and iso in STRAE_MOMENTS:
            probed = STRAE_MOMENTS[iso]
            score = strae(place(pred, probed["mean"], probed["sd"]), y, lo, hi, w)
            print(f"{'board-probed':<26}{probed['mean']:>7.2f}{probed['sd']:>7.2f}{score:>10.4f}")


if __name__ == "__main__":
    main()
