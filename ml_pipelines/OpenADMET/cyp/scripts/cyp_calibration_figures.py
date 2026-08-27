"""The placement figure for `docs/blogs/cyp_challenge.md`.

Writes `docs/images/cyp_calibration_applied.svg` -- our raw blind-set predictions, the
same predictions after placement, and the blind population they are scored against.

That population is drawn as a curve rather than a histogram because its labels are hidden;
only its mean and sd are known, solved from scored submissions.

    python cyp_calibration_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cyp_recalibrate import BLIND_MOMENTS, SOLVED_PEARSON, VALUE_COLUMNS

IMAGES = Path(__file__).resolve().parents[4] / "docs" / "images"
OUT = Path(__file__).parent / "outputs"
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

TRAIN = "#4C72B0"
BLIND = "#C44E52"
CALIB = "#55A868"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def normal(ax, mean, sd, color, label, lo=1.0, hi=9.0):
    x = np.linspace(lo, hi, 400)
    ax.plot(x, np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=color, lw=2.5, label=label)


def figure_applied(raw: pd.DataFrame, placed: pd.DataFrame) -> None:
    """Our own blind-set predictions, before and after placement."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, iso in zip(axes.ravel(), ISOFORMS):
        col = VALUE_COLUMNS[iso]
        ax.hist(raw[col], bins=35, density=True, color=TRAIN, alpha=0.5, label="raw predictions")
        ax.hist(placed[col], bins=35, density=True, color=CALIB, alpha=0.5, label="after placement")
        m, sd = BLIND_MOMENTS[iso]["mean"], BLIND_MOMENTS[iso]["sd"]
        normal(ax, m, sd, BLIND, "blind population")
        ax.set_title(
            f"{iso}   sd {raw[col].std():.2f} → {placed[col].std():.2f}"
            f"   (target ρ·sd = {SOLVED_PEARSON[iso] * sd:.2f})"
        )
        ax.set_xlabel("pIC50")
        ax.set_ylabel("density")
        ax.set_xlim(1, 9)
        ax.legend(loc="upper left", framealpha=0.9)
    fig.suptitle("The same predictions, moved onto the population they are scored against", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(IMAGES / "cyp_calibration_applied.svg")
    plt.close(fig)


if __name__ == "__main__":
    raw = pd.read_csv(OUT / "cyp-reg-chemprop-mt-aux-100_activity_submission.csv")
    placed = pd.read_csv(OUT / "cyp-reg-chemprop-mt-aux-100_activity_submission_solved.csv")

    figure_applied(raw, placed)
    print(f"Wrote cyp_calibration_applied.svg to {IMAGES}")
