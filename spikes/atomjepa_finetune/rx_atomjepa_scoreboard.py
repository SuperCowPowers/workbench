"""ExpansionRx scoreboard: fine-tuned Atom-JEPA against an assay's other models, out of fold.

Every model is scored on its out-of-fold predictions over the assay's FeatureSet rows: the
deployed models from their full_cross_fold capture, the endpoint-less Atom-JEPA spike from
its "training" capture. Folds differ between models, rows and units do not. Deltas are
paired MAE with a 95% bootstrap interval (negative = better than chemprop).

Pat Walters' independent run of the authors' code (Model Validation Central, atom-jepa
report; multi-task over all 9 endpoints, 5x5 CV, same units) for reference, MAE:

    assay   Atom-JEPA FT   CheMeleon FT   Monroe + TabPFN 3
    mppb           0.185          0.212               0.259
    logd           0.382          0.427               0.402

    python rx_atomjepa_scoreboard.py mppb
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from workbench.api import Model
from workbench.utils.metrics_utils import bootstrap_compare

id_col = "molecule_name"
FRAMEWORKS = ["chemprop", "chemprop-desc", "chemprop-hpo", "xgb", "pytorch"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("assay")
args = parser.parse_args()
target = args.assay
prefix = target.replace("_", "-")


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


def oof(model_name: str, capture: str) -> pd.DataFrame:
    """A model's out-of-fold predictions, indexed by molecule_name."""
    frame = Model(model_name).get_inference_predictions(capture).set_index(id_col)[[target, "prediction"]]
    return frame.dropna().sort_index()


frames = {"atomjepa": oof(f"{prefix}-reg-atomjepa", "training")}
frames.update({fw: oof(f"{prefix}-reg-{fw}", "full_cross_fold") for fw in FRAMEWORKS})
ids = sorted(set.intersection(*(set(f.index) for f in frames.values())))
frames = {arm: frame.loc[ids] for arm, frame in frames.items()}

rows = []
for arm, frame in frames.items():
    y, p = frame[target], frame["prediction"]
    row = {
        "model": arm,
        "mae": mae(frame),
        "spearman": spearmanr(y, p).statistic,
        "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
    }
    if arm != "chemprop":
        cmp = bootstrap_compare(frame, frames["chemprop"], mae)
        row["mae vs chemprop"] = f"{cmp['delta']:+.3f} [{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]"
    rows.append(row)

print(f"{target}, out of fold on {len(ids)} compounds (MAE: lower is better)")
print(pd.DataFrame(rows).sort_values("mae").fillna("").to_string(index=False, float_format="%.3f"))
