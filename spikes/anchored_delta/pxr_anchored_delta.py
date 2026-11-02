"""PXR anchored-delta spike: correct each prediction with its nearest labeled analogs' residuals.

    y_hat(x) = pred(x) + alpha * sum_topk(s_j * r_j) / sum_topk(s_j)

s_j is Tanimoto similarity (count Morgan) to anchor j and r_j = y_j - pred_j its residual. This is
the useful core of Causal-Chemprop (neopolyai/causal_chemprop): its counterfactual keeps the anchor's
noise term, so it amounts to anchoring on a measured analog.

(k, alpha > 0) are chosen on the training OOF (leave-one-out over the anchor pool) and applied once
to the 253 held-out phase1_test compounds, with two anchor pools:
  - train:  OOF residuals of the training compounds (usable for a blind submission)
  - series: the other held-out compounds' residuals, leave-one-out (the paper's setting: part of
            the new series is already measured)
The test-tuned row is the optimistic number you get by sweeping (k, alpha) on the held-out itself.

Run from anywhere:  python pxr_anchored_delta.py
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs
from scipy.stats import spearmanr

from workbench.api import FeatureSet, Model
from workbench.utils.chem_utils.fingerprints import similarity_fingerprints
from workbench.utils.metrics_utils import bootstrap_compare

id_col, target = "molecule_name", "pec50"
MODELS = ["pxr-reg-chemprop-phase1", "pxr-reg-tabicl-monroe-readout-phase1"]
KS = [1, 3, 5, 10, 25]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SIM_BINS = [0.0, 0.4, 0.6, 1.01]


def rae(frame: pd.DataFrame) -> float:
    """Relative absolute error against the mean predictor (the challenge's headline)."""
    y = frame[target]
    return float((frame["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


def similarity(query_smiles: list[str], anchor_smiles: list[str]) -> np.ndarray:
    """Tanimoto similarity matrix, queries x anchors."""
    q_fps, q_pos = similarity_fingerprints(query_smiles)
    a_fps, a_pos = similarity_fingerprints(anchor_smiles)
    assert len(q_pos) == len(query_smiles) and len(a_pos) == len(anchor_smiles), "unparseable SMILES"
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, a_fps) for fp in q_fps])


def top_neighbors(sims: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and similarities of each query's k most similar anchors, most similar first."""
    idx = np.argsort(-sims, axis=1)[:, :k]
    return idx, np.take_along_axis(sims, idx, axis=1)


def correction(neighbors: tuple[np.ndarray, np.ndarray], residuals: np.ndarray, k: int) -> np.ndarray:
    """Similarity-weighted mean residual of each query's top-k anchors."""
    idx, s = neighbors[0][:, :k], neighbors[1][:, :k]
    return (s * residuals[idx]).sum(axis=1) / np.clip(s.sum(axis=1), 1e-9, None)


def sweep(frame: pd.DataFrame, neighbors, residuals: np.ndarray, alphas=ALPHAS) -> tuple[float, int, float]:
    """Best (rae, k, alpha) over the grid; alpha=0 is the uncorrected baseline."""
    grid = []
    for k in KS:
        c = correction(neighbors, residuals, k)
        grid += [(rae(frame.assign(prediction=frame["prediction"] + a * c)), k, a) for a in alphas]
    return min(grid)


smiles = FeatureSet("openadmet_pxr_f1").pull_dataframe().set_index(id_col)["smiles"]

for name in MODELS:
    model = Model(name)
    held = model.get_inference_predictions("pxr_phase1_test").set_index(id_col).sort_index()[[target, "prediction"]]
    oof = model.get_inference_predictions("full_cross_fold").set_index(id_col)[[target, "prediction"]]
    anchors = oof.drop(index=held.index, errors="ignore")  # a capture may also score the held-out rows
    r_anchor = (anchors[target] - anchors["prediction"]).to_numpy()
    r_held = (held[target] - held["prediction"]).to_numpy()
    anchor_smiles, held_smiles = smiles.loc[anchors.index].tolist(), smiles.loc[held.index].tolist()

    # Select (k, alpha > 0) on training OOF, leave-one-out over the anchor pool
    s_aa = similarity(anchor_smiles, anchor_smiles)
    np.fill_diagonal(s_aa, -1.0)
    train_neighbors = top_neighbors(s_aa, max(KS))
    del s_aa
    train_rae, k, alpha = sweep(anchors, train_neighbors, r_anchor, alphas=ALPHAS[1:])
    train_resid_corr = spearmanr(correction(train_neighbors, r_anchor, k), r_anchor)[0]

    s_ha = similarity(held_smiles, anchor_smiles)
    s_hh = similarity(held_smiles, held_smiles)
    np.fill_diagonal(s_hh, -1.0)
    pools = {"train": (top_neighbors(s_ha, max(KS)), r_anchor), "series": (top_neighbors(s_hh, max(KS)), r_held)}

    print(f"\n=== {name}: {len(anchors)} anchors, {len(held)} held-out")
    print(
        f"train OOF (LOO): RAE {rae(anchors):.3f} -> {train_rae:.3f} at k={k}, alpha={alpha}"
        f" (resid_corr {train_resid_corr:.3f})"
    )

    rows = [
        {
            "arm": "baseline",
            "rae": rae(held),
            "mae": mae(held),
            "spearman": spearmanr(held[target], held["prediction"])[0],
        }
    ]
    for pool, (neighbors, residuals) in pools.items():
        c = correction(neighbors, residuals, k)
        corrected = held.assign(prediction=held["prediction"] + alpha * c)
        cmp = bootstrap_compare(corrected, held, rae)
        rows.append(
            {
                "arm": f"{pool} anchors",
                "rae": rae(corrected),
                "mae": mae(corrected),
                "spearman": spearmanr(corrected[target], corrected["prediction"])[0],
                "resid_corr": spearmanr(c, r_held)[0],
                "vs baseline": f"{cmp['delta']:+.3f} [{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]",
            }
        )
        tuned_rae, tuned_k, tuned_a = sweep(held, neighbors, residuals)
        rows.append({"arm": f"{pool} test-tuned (k={tuned_k}, a={tuned_a})", "rae": tuned_rae})
    print(pd.DataFrame(rows).round(3).fillna("").to_string(index=False))

    # Error by similarity to the nearest training anchor
    c_train = correction(pools["train"][0], r_anchor, k)
    by_sim = pd.DataFrame(
        {
            "max_sim": pd.cut(s_ha.max(axis=1), SIM_BINS, right=False),
            "base_err": np.abs(r_held),
            "corr_err": np.abs(r_held - alpha * c_train),
        }
    )
    print(
        by_sim.groupby("max_sim", observed=True)
        .agg(n=("base_err", "size"), base_mae=("base_err", "mean"), corr_mae=("corr_err", "mean"))
        .to_string(float_format="{:.3f}".format)
    )
