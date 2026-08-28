"""What difference can each ruler actually resolve?

Every candidate comparison we make is two models scored on one row set, so the noise has
two independent parts:

  row sampling -- which compounds happen to be in the set. Bootstrapped here from the
      *paired* difference, because both models see the same rows and most of this cancels.
  training stochasticity -- which fold split and weight init a run drew. Estimated from
      replicates of one config at different seeds, and it does NOT cancel: two candidates
      are two independent draws.

A difference is resolvable at roughly 2 sigma of their sum. Row sampling shrinks as
1/sqrt(n) but also as (1 - rho^2), so a ruler on which the model already correlates well is
quieter at the same size -- transporting a noise estimate between rulers has to carry both
terms. Training noise shrinks only by averaging k seeds per arm.

The board is therefore not the weak ruler its 375 rows suggest: our board Spearman runs far
above our OOF Spearman, and the two effects nearly cancel.

    python cyp_ruler_power.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cyp_seed_noise import oof

ISOFORMS = ["cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4"]
SEEDS = ["cyp-reg-chemprop-mt-aux-100", "cyp-reg-chemprop-mt-aux-100-s43", "cyp-reg-chemprop-mt-aux-100-s44"]
PAIR = ("cyp-reg-chemprop-mt-aux-100", "cyp-reg-chemprop-union-p30")
BOARD_N = 375          # the live half
# Spearman on each ruler, needed because sampling noise scales with (1 - rho^2). OOF is the
# control model's; board is the mean of the two entries being compared.
RHO_OOF = {"cyp1a2": 0.556, "cyp2c9": 0.683, "cyp2d6": 0.436, "cyp3a4": 0.802}
RHO_BOARD = {"cyp1a2": 0.807, "cyp2c9": 0.843, "cyp2d6": 0.455, "cyp3a4": 0.843}
N_BOOT = 400
RNG = np.random.default_rng(0)


def paired_sampling_sd(iso: str) -> tuple[float, int]:
    """Bootstrap sd of the paired Spearman difference between two configs, at OOF's n."""
    target = f"{iso}_pic50_direct_inhibition"
    a, b = (oof(m, iso) for m in PAIR)
    j = a[[target, "prediction"]].join(b[["prediction"]], rsuffix="_b", how="inner").dropna()
    y, pa, pb = j[target].to_numpy(), j["prediction"].to_numpy(), j["prediction_b"].to_numpy()
    n = len(j)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        diffs[i] = spearmanr(y[idx], pb[idx]).statistic - spearmanr(y[idx], pa[idx]).statistic
    return float(diffs.std()), n


def seed_sd(iso: str) -> float:
    """Sd of Spearman across replicates of one config — the part that never cancels."""
    target = f"{iso}_pic50_direct_inhibition"
    vals = []
    for m in SEEDS:
        d = oof(m, iso)
        vals.append(spearmanr(d[target], d["prediction"]).statistic)
    return float(np.std(vals, ddof=1))


if __name__ == "__main__":
    rows = []
    for iso in ISOFORMS:
        samp, n_oof = paired_sampling_sd(iso)
        seed = seed_sd(iso)
        # Carry both terms: sqrt(n) for set size, (1 - rho^2) for where each ruler sits.
        board_samp = (samp * np.sqrt(n_oof / BOARD_N)
                      * (1 - RHO_BOARD[iso] ** 2) / (1 - RHO_OOF[iso] ** 2))
        rows.append({
            "isoform": iso,
            "n_oof": n_oof,
            "row sd @OOF": samp,
            "row sd @board": board_samp,
            "seed sd": seed,
            # one build per arm: both noise sources; k seeds per arm: seed part /sqrt(k)
            "OOF 1x": 2 * np.hypot(samp, seed * np.sqrt(2)),
            "OOF 3x": 2 * np.hypot(samp, seed * np.sqrt(2 / 3)),
            "board 1x": 2 * np.hypot(board_samp, seed * np.sqrt(2)),
        })
    df = pd.DataFrame(rows).set_index("isoform")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())
    print("\nSmallest resolvable Spearman difference (2 sigma):")
    print(df[["OOF 1x", "OOF 3x", "board 1x"]].round(3).to_string())
