"""How the CYP ensemble combines its members, in one place.

Both the submission builder and the placement step need the same combination: the
submission writes it out, and the placement step needs its Pearson to decide the spread.
A second implementation in either would drift from the other, and the placement would then
describe a vector nobody submitted.

The combination is two steps. Members are weighted per compound by confidence scaled by
each one's |conf-to-error correlation|, then that correction is extrapolated:

    base = sum(pred * conf_weights_with_fallback(conf * corr_scale, fallback_w))
    out  = base + lam * (base - pred.mean())

`base - flat` is the confidence correction; out of fold it is under-applied by a factor of
2.5 to 3.5, and `lam` restores it. The second step combines two different vectors rather
than rescaling one, so it reorders -- an affine rescale could not move Spearman at all.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from workbench.api import FeatureSet, Model
from workbench.utils.ensemble_simulator import EnsembleSimulator
from workbench.utils.ensemble_utils import conf_weights_with_fallback
from workbench.utils.metrics_utils import soft_threshold_rae

ID_COLUMN = "molecule_name"
OOF_FS = "openadmet_cyp_aux_f1"
LAMBDA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

MT = "cyp-reg-chemprop-union-p30"
AUX = "cyp-reg-chemprop-mt-aux-100"
TOX = "cyp-reg-chemprop-union-p30-tox"


def tabicl(iso: str) -> str:
    """TabICL over the Monroe embedding, one model per isoform."""
    return f"cyp-reg-tabicl-{iso.lower().removeprefix('cyp')}-monroe"


MEMBERS = {
    "CYP1A2": [MT, AUX, TOX, tabicl("CYP1A2")],
    "CYP2C9": [MT, AUX, TOX, tabicl("CYP2C9")],
    "CYP2D6": [MT, AUX, TOX, tabicl("CYP2D6"), "cyp-reg-chemprop-2d6-isoform", "cyp-reg-chemprop-2d6-single"],
    "CYP3A4": [MT, AUX, TOX, tabicl("CYP3A4")],
}


def combine(pred: np.ndarray, conf: np.ndarray, corr_scale: np.ndarray, fallback_w: np.ndarray, lam: float):
    """The ensemble's prediction for rows laid out as (n_rows, n_members)."""
    base = (pred * conf_weights_with_fallback(conf * corr_scale, fallback_w)).sum(axis=1)
    return base + lam * (base - pred.mean(axis=1))


def member_scaling(iso: str, members: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-member `corr_scale` and fallback weights, read off the out-of-fold captures.

    Returns both arrays in `members` order. `corr_scale` is each member's
    |conf-to-error correlation|; the fallback weights are inverse-MAE, used on any row
    whose confidences carry no usable signal.
    """
    sim = EnsembleSimulator(members, id_column=ID_COLUMN, target=f"{iso.lower()}_pic50_direct_inhibition")
    config = sim.get_best_strategy_config(select_by="spearman")
    if config["aggregation_strategy"] != "calibrated_conf_weighted":
        print(
            f"  WARNING {iso}: out-of-fold now prefers '{config['aggregation_strategy']}' "
            f"over calibrated_conf_weighted; this script still applies the latter"
        )
    return (
        np.array([config["corr_scale"][m] for m in members]),
        np.array([config["model_weights"][m] for m in members]),
    )


def member_frame(iso: str, members: list[str]) -> pd.DataFrame:
    """Out-of-fold prediction and confidence per member, with label and credible interval.

    Columns are `<model>__pred`, `<model>__conf`, `label`, `lo`, `hi`, indexed by compound
    over the rows every member scored. Intervals are label metadata that captures trim, so
    they are joined back from the FeatureSet by id.
    """
    target = f"{iso.lower()}_pic50_direct_inhibition"
    cols = {}
    for name in members:
        model = Model(name)
        declared = model.target()
        declared = list(declared) if isinstance(declared, list) else [declared]
        capture = f"cv_{target}" if len(declared) > 1 else "full_cross_fold"
        d = model.get_inference_predictions(capture).set_index(ID_COLUMN)
        cols[f"{name}__pred"] = d["prediction"]
        cols[f"{name}__conf"] = d["confidence"]
        cols["label"] = d[target]
    frame = pd.DataFrame(cols)
    bounds = FeatureSet(OOF_FS).pull_dataframe().set_index(ID_COLUMN)
    bounds = bounds[[f"{target}_ci_lower", f"{target}_ci_upper"]]
    frame = frame.join(bounds.rename(columns={f"{target}_ci_lower": "lo", f"{target}_ci_upper": "hi"}))
    return frame.dropna().sort_index()


def oof_arrays(iso: str, members: list[str]):
    """`(y, lo, hi, pred, conf)` for an isoform, aligned over the rows every member scored."""
    frame = member_frame(iso, members)
    return (
        frame["label"].to_numpy(float),
        frame["lo"].to_numpy(float),
        frame["hi"].to_numpy(float),
        frame[[f"{m}__pred" for m in members]].to_numpy(float),
        frame[[f"{m}__conf" for m in members]].to_numpy(float),
    )


def fit_lambda(iso: str, members: list[str], corr_scale: np.ndarray, fallback_w: np.ndarray, verbose: bool = True):
    """How far past the weighted combination to extrapolate, chosen out of fold.

    Picked on held-out hinge over `LAMBDA_GRID`, taking the median across five folds.
    """
    y, lo, hi, pred, conf = oof_arrays(iso, members)
    base = combine(pred, conf, corr_scale, fallback_w, 0.0)
    flat = pred.mean(axis=1)

    picks = []
    for train_idx, _ in KFold(n_splits=5, shuffle=True, random_state=42).split(base):
        picks.append(
            min(
                LAMBDA_GRID,
                key=lambda lam: soft_threshold_rae(
                    y[train_idx], (base + lam * (base - flat))[train_idx], lo[train_idx], hi[train_idx]
                ),
            )
        )
    lam = float(np.median(picks))
    if verbose:
        held = base + lam * (base - flat)
        print(
            f"  {iso}: lambda {lam:g} (folds {picks}), out-of-fold spearman "
            f"{spearmanr(y, base).statistic:.4f} -> {spearmanr(y, held).statistic:.4f}, "
            f"ST-RAE {soft_threshold_rae(y, base, lo, hi):.4f} -> {soft_threshold_rae(y, held, lo, hi):.4f}"
        )
    return lam
