"""Does a candidate model earn a slot in an isoform's ensemble pool?

Membership is decided by decorrelation, not by solo score, so this reports both: each
member's own out-of-fold Pearson, how correlated its predictions are with the rest of
the pool, and what the pool's Pearson does when the candidate joins.

Read the delta against `cyp_ruler_power.py` thresholds. A candidate that lifts the pool
by less than OOF can resolve has not earned anything, however good its solo number looks.

    python cyp_member_diversity.py --candidate cyp-fp-reg-xgb
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from cyp_ensemble_submit import MEMBERS
from scipy.stats import pearsonr
from workbench.api import Model

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
# Smallest OOF-resolvable Pearson difference per isoform (scripts/cyp_ruler_power.py).
RESOLUTION = {"CYP1A2": 0.043, "CYP2C9": 0.031, "CYP2D6": 0.056, "CYP3A4": 0.018}


def oof(model_name: str, target: str) -> pd.DataFrame | None:
    """Out-of-fold truth and prediction for one model on one target, indexed by compound."""
    model = Model(model_name)
    runs = model.list_inference_runs()
    # Single-target models report under `full_cross_fold`; multi-target under `cv_<target>`.
    run = f"cv_{target}" if f"cv_{target}" in runs else "full_cross_fold"
    if run not in runs:
        return None
    df = model.get_inference_predictions(run)
    if df is None or target not in df.columns:
        return None
    return df[["molecule_name", target, "prediction"]].dropna().set_index("molecule_name")


def pool_pearson(frames: dict, members: list, rows: list, truth: pd.Series) -> float:
    """Pearson of the averaged predictions of `members` over `rows`."""
    stacked = np.column_stack([frames[m].loc[rows, "prediction"] for m in members])
    return pearsonr(truth, stacked.mean(axis=1)).statistic


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", required=True, help="model name, or a prefix for per-isoform models")
args = parser.parse_args()

for iso in ISOFORMS:
    target = f"{iso.lower()}_pic50_direct_inhibition"
    # Per-isoform model families are suffixed with the bare isoform (3a4, 2c9, ...).
    candidate = args.candidate
    if oof(candidate, target) is None:
        candidate = f"{args.candidate}-{iso.lower().removeprefix('cyp')}"

    names = MEMBERS[iso] + [candidate]
    frames = {n: oof(n, target) for n in names}
    missing = [n for n, f in frames.items() if f is None]
    if missing:
        print(f"\n{iso}: no out-of-fold capture for {', '.join(missing)} — skipped")
        continue

    rows = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    truth = frames[names[0]].loc[rows, target]
    print(f"\n=== {iso} ({len(rows):,} shared rows) ===")

    print(f"{'member':<34}{'solo r':>8}")
    for n in names:
        solo = pearsonr(truth, frames[n].loc[rows, "prediction"]).statistic
        print(f"{n:<34}{solo:>8.3f}{'   <- candidate' if n == candidate else ''}")

    print(f"\n{'pair':<34}{'pred r':>8}")
    for a, b in combinations(names, 2):
        r = pearsonr(frames[a].loc[rows, "prediction"], frames[b].loc[rows, "prediction"]).statistic
        tag = "   <- vs candidate" if candidate in (a, b) else ""
        print(f"{a.split('-')[-1]} / {b.split('-')[-1]:<24}{r:>8.3f}{tag}")

    base = pool_pearson(frames, MEMBERS[iso], rows, truth)
    joined = pool_pearson(frames, names, rows, truth)
    verdict = "earns a slot" if joined - base > RESOLUTION[iso] else "inside noise"
    print(f"\npool {base:.3f} -> {joined:.3f} ({joined - base:+.3f}, resolves at {RESOLUTION[iso]:.3f}) {verdict}")
