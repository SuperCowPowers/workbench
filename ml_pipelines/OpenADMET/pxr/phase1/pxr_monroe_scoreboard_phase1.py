"""PXR phase-1 scoreboard: the deployed TabICL + Monroe + readouts model against Chemprop.

Every arm is a deployed model's `pxr_phase1_test` capture (253 held-out compounds, never
trained on), paired by molecule_name. RAE is the challenge's headline; deltas are paired
RAE with a 95% bootstrap interval (negative = better than the comparison).

Runs locally (not via ml_pipeline_launcher):  python pxr_monroe_scoreboard_phase1.py
"""

import pandas as pd

from workbench.api import Model
from workbench.utils.metrics_utils import bootstrap_compare

id_col, target = "molecule_name", "pec50"
ARMS = {
    "chemprop": "pxr-reg-chemprop-phase1",
    "chemprop_readout": "pxr-reg-chemprop-readout-phase1",
    "tabicl_monroe_readout": "pxr-reg-tabicl-monroe-readout-phase1",
}


def rae(frame: pd.DataFrame) -> float:
    """Relative absolute error against the mean predictor (the challenge's headline)."""
    y = frame[target]
    return float((frame["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


def captured(model_name: str) -> pd.DataFrame:
    """A deployed model's held-out capture, indexed by molecule_name."""
    capture = Model(model_name).get_inference_predictions("pxr_phase1_test")
    return capture.set_index(id_col)[[target, "prediction"]].sort_index()


frames = {arm: captured(model_name) for arm, model_name in ARMS.items()}
ids = frames["chemprop"].index
assert all(frame.index.equals(ids) for frame in frames.values()), "captures cover different molecules"

rows = []
for arm, frame in frames.items():
    row = {"arm": arm, "rae": rae(frame), "mae": mae(frame)}
    for base in ("chemprop", "chemprop_readout"):
        if base != arm and list(ARMS).index(base) < list(ARMS).index(arm):
            cmp = bootstrap_compare(frame, frames[base], rae)
            row[f"vs {base}"] = f"{cmp['delta']:+.3f} [{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]"
    rows.append(row)

print(f"phase1_test, {len(ids)} compounds (RAE: lower is better)")
print(pd.DataFrame(rows).fillna("").to_string(index=False, float_format="%.3f"))
