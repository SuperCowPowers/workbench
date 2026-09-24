"""PXR phase-1 spike: TabICL on CheMeleon embeddings + primary-screen readouts.

Tests the OpenADMET "Are tabular foundation models all you need?" recipe: a CheMeleon
embedding plus predicted primary-screen log2FC readouts, fed to TabICL (BSD-3 code and
weights; TabPFN >= 2.5 weights are non-commercial so it is not considered).

Runs locally (not via ml_pipeline_launcher) against openadmet_pxr_readout: fit on the
`train` rows, score on the 253 `phase1_test` rows, every arm paired against
`pxr-reg-chemprop-phase1` on the same rows.

Arms:
  chemprop          pxr-reg-chemprop-phase1's `pxr_phase1_test` capture (the bar)
  chemprop_readout  pxr-reg-chemprop-readout-phase1's capture (chemprop + readouts)
  tabicl            CheMeleon PCA -> TabICL
  tabicl_readout    CheMeleon PCA + readouts -> TabICL (the article's recipe)

TabICL arms are averaged over SEEDS before the paired bootstrap, so the comparison
carries row noise only; the per-seed spread is reported alongside.

Without the readouts TabICL loses (RAE 0.687 vs chemprop 0.591, +0.096 [+0.056, +0.142]),
and stacking chemprop's own OOF pEC50 prediction only recovers chemprop (0.609,
+0.019 [-0.004, +0.043]).

Needs openadmet_pxr_readout and pxr-reg-chemprop-readout-phase1 (see pipelines.json).

    uv run --with tabicl python pxr_tabicl_spike_phase1.py
"""

import numpy as np
import pandas as pd
import torch
from chemprop import data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor

from workbench.api import FeatureSet, Model
from workbench.training.chemprop_core import load_foundation_weights
from workbench.utils.metrics_utils import bootstrap_compare

fs_name = "openadmet_pxr_readout"
baseline_model = "pxr-reg-chemprop-phase1"
readout_model = "pxr-reg-chemprop-readout-phase1"
id_col, target = "molecule_name", "pec50"
READOUTS = ["lfc_8um_readout", "lfc_33um_readout"]
PCA_DIMS = 256
SEEDS = [0, 1, 2, 3, 4]


def chemeleon_embed(smiles: list[str], batch_size: int = 256) -> np.ndarray:
    """Frozen CheMeleon MPNN + mean aggregation -> one 2048-d vector per molecule."""
    mp, agg = load_foundation_weights("CheMeleon")
    mp.eval()
    dataset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
    loader = data.build_dataloader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(agg(mp(batch.bmg), batch.bmg.batch).numpy())
    return np.vstack(chunks)


def rae(frame: pd.DataFrame) -> float:
    """Relative absolute error against the mean predictor (the challenge's headline)."""
    y = frame[target]
    return float((frame["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


# Data: train rows fit, phase1_test rows score
df = FeatureSet(fs_name).pull_dataframe()[[id_col, "smiles", target, "split"] + READOUTS]
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "phase1_test"].reset_index(drop=True)
print(f"{len(train)} train / {len(test)} phase1_test rows")

# CheMeleon embedding -> standardize -> PCA, both fit on train only
emb_train, emb_test = chemeleon_embed(train["smiles"].tolist()), chemeleon_embed(test["smiles"].tolist())
scaler = StandardScaler().fit(emb_train)
pca = PCA(n_components=PCA_DIMS, random_state=0).fit(scaler.transform(emb_train))
X_train, X_test = pca.transform(scaler.transform(emb_train)), pca.transform(scaler.transform(emb_test))
print(f"PCA {emb_train.shape[1]} -> {PCA_DIMS}: {pca.explained_variance_ratio_.sum():.1%} variance kept")
X_train_ro = np.column_stack([X_train, train[READOUTS].to_numpy()])
X_test_ro = np.column_stack([X_test, test[READOUTS].to_numpy()])


def tabicl(seed: int, X_fit: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """CPU: on MPS a single 4k-row fit asks for ~8 GiB of shared GPU memory."""
    return TabICLRegressor(random_state=seed, device="cpu").fit(X_fit, train[target]).predict(X_pred)


def frame(pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({target: test[target].to_numpy(), "prediction": pred}, index=test[id_col])


def captured(model_name: str) -> np.ndarray:
    """A deployed model's held-out `pxr_phase1_test` predictions, aligned to `test`."""
    held = Model(model_name).get_inference_predictions("pxr_phase1_test").set_index(id_col)["prediction"]
    pred = test[id_col].map(held).to_numpy()
    assert not np.isnan(pred).any(), f"{model_name} capture is missing phase1_test rows"
    return pred


# Arms: name -> per-seed predictions (captured models are a single "seed")
arms = {
    "chemprop_readout": lambda: np.array([captured(readout_model)]),
    "tabicl": lambda: np.array([tabicl(seed, X_train, X_test) for seed in SEEDS]),
    "tabicl_readout": lambda: np.array([tabicl(seed, X_train_ro, X_test_ro) for seed in SEEDS]),
}

base = frame(captured(baseline_model))
rows = [{"arm": "chemprop", "rae": rae(base), "mae": mae(base), "seed_sd": np.nan, "delta": 0.0, "ci": ""}]
for name, predict in arms.items():
    per_seed = predict()
    pred = frame(per_seed.mean(axis=0))
    cmp = bootstrap_compare(pred, base, rae)
    rows.append(
        {
            "arm": name,
            "rae": cmp["value_a"],
            "mae": mae(pred),
            "seed_sd": float(np.std([rae(frame(p)) for p in per_seed])) if len(per_seed) > 1 else np.nan,
            "delta": cmp["delta"],
            "ci": f"[{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]",
            "p_better": cmp["p_a_better"],
        }
    )
    print(f"{name}: RAE {cmp['value_a']:.3f} (delta vs chemprop {cmp['delta']:+.3f})")

results = pd.DataFrame(rows)
print("\nphase1_test RAE (lower is better); delta/ci/p_better are paired RAE vs chemprop")
print(results.to_string(index=False, float_format="%.3f"))
