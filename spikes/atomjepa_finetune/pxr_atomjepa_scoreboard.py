"""PXR phase-1 scoreboard: fine-tuned Atom-JEPA against Chemprop and TabICL + Monroe.

Every arm is scored on the 253 held-out phase1_test compounds (never trained on), paired by
molecule_name. The deployed models are read from their `pxr_phase1_test` endpoint capture;
the Atom-JEPA spike has no endpoint, so its held-out rows come from the "validation"
capture its training job wrote. RAE is the challenge's headline; deltas are paired RAE with
a 95% bootstrap interval (negative = better than the comparison).

Run from anywhere:  python pxr_atomjepa_scoreboard.py
"""

import pandas as pd

from workbench.api import Model
from workbench.utils.metrics_utils import bootstrap_compare

id_col, target = "molecule_name", "pec50"
# arm -> (model, capture)
ARMS = {
    "chemprop": ("pxr-reg-chemprop-phase1", "pxr_phase1_test"),
    "chemprop_readout": ("pxr-reg-chemprop-readout-phase1", "pxr_phase1_test"),
    "tabicl_monroe_readout": ("pxr-reg-tabicl-monroe-readout-phase1", "pxr_phase1_test"),
    "atomjepa_readout": ("pxr-reg-atomjepa-readout-phase1", "validation"),
}
BASES = ["chemprop", "chemprop_readout", "tabicl_monroe_readout"]


def rae(frame: pd.DataFrame) -> float:
    """Relative absolute error against the mean predictor (the challenge's headline)."""
    y = frame[target]
    return float((frame["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


def captured(model_name: str, capture: str) -> pd.DataFrame:
    """A model's held-out predictions, indexed by molecule_name."""
    frame = Model(model_name).get_inference_predictions(capture).set_index(id_col)[[target, "prediction"]]
    assert frame["prediction"].notna().all(), f"{model_name} {capture} has rows without a prediction"
    return frame.sort_index()


frames = {arm: captured(*source) for arm, source in ARMS.items()}
ids = frames["chemprop"].index
assert all(frame.index.equals(ids) for frame in frames.values()), "captures cover different molecules"

rows = []
for arm, frame in frames.items():
    row = {"arm": arm, "rae": rae(frame), "mae": mae(frame)}
    for base in BASES:
        if list(ARMS).index(base) < list(ARMS).index(arm):
            cmp = bootstrap_compare(frame, frames[base], rae)
            row[f"vs {base}"] = f"{cmp['delta']:+.3f} [{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]"
    rows.append(row)

print(f"phase1_test, {len(ids)} compounds (RAE: lower is better)")
print(pd.DataFrame(rows).fillna("").to_string(index=False, float_format="%.3f"))
