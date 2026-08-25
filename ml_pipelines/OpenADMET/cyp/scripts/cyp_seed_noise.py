"""Report the seed-to-seed spread of OOF metrics — the floor a candidate delta must clear.

Reads the `cv_*` captures from replicates of one config and reports, per isoform, the
spread of Pearson and Spearman across them. A candidate that moves an isoform by less than
its spread here has not been shown to do anything.

Also reports how well two replicates agree on individual compounds. Low agreement with
similar aggregate metrics means the replicates are making different mistakes, which is
what makes ensembling them worth trying.

    python cyp_seed_noise.py
    python cyp_seed_noise.py --models modelA modelB modelC
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from workbench.api import Model

ISOFORMS = ["cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4"]
DEFAULT_MODELS = [
    "cyp-reg-chemprop-mt-aux-100",
    "cyp-reg-chemprop-mt-aux-100-s43",
    "cyp-reg-chemprop-mt-aux-100-s44",
]


def oof(model_name: str, iso: str) -> pd.DataFrame:
    """Out-of-fold predictions for one isoform, indexed by compound."""
    target = f"{iso}_pic50_direct_inhibition"
    model = Model(model_name)
    run = f"cv_{target}"
    if run not in model.list_inference_runs():
        raise ValueError(f"'{model_name}' has no capture '{run}'")
    d = model.get_inference_predictions(run)
    id_col = "molecule_name" if "molecule_name" in d.columns else d.columns[0]
    return d[[id_col, target, "prediction"]].dropna().set_index(id_col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Replicates of one config")
    args = parser.parse_args()

    print(f"{len(args.models)} replicates: {', '.join(args.models)}\n")
    print(f"{'isoform':<8}{'n':>6}{'pearson (each)':>34}{'spread':>9}{'spearman spread':>17}")
    for iso in ISOFORMS:
        frames = {m: oof(m, iso) for m in args.models}
        target = f"{iso}_pic50_direct_inhibition"
        pear = [pearsonr(d[target], d["prediction"]).statistic for d in frames.values()]
        spear = [spearmanr(d[target], d["prediction"]).statistic for d in frames.values()]
        n = min(len(d) for d in frames.values())
        each = " ".join(f"{p:.4f}" for p in pear)
        print(f"{iso:<8}{n:>6}{each:>34}{max(pear) - min(pear):>9.4f}{max(spear) - min(spear):>17.4f}")

    print(f"\n{'isoform':<8}{'pairwise agreement between replicate predictions':>52}")
    for iso in ISOFORMS:
        frames = {m: oof(m, iso) for m in args.models}
        agree = []
        for a, b in combinations(args.models, 2):
            joined = frames[a][["prediction"]].join(frames[b][["prediction"]], how="inner", rsuffix="_b")
            agree.append(pearsonr(joined["prediction"], joined["prediction_b"]).statistic)
        print(f"{iso:<8}{np.mean(agree):>52.4f}")
