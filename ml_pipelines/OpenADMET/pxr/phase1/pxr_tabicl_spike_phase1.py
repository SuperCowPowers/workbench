"""PXR phase-1 spike: TabICL on frozen CheMeleon embeddings — NEGATIVE result (kept for the record).

Tests the OpenADMET "Are tabular foundation models all you need?" recipe (TabICL,
BSD-3 code and weights; TabPFN >= 2.5 weights are non-commercial so it was not
considered). Held-out phase1_test RAE, paired bootstrap vs chemprop:

    chemprop    0.591
    tabicl      0.687   +0.096 [+0.056, +0.142]  clearly worse
    tabicl_oof  0.609   +0.019 [-0.004, +0.043]  no gain; chemprop wins 94.5% of resamples

Seed spread is 0.003-0.004, so these are model differences, not noise. TabICL on
frozen CheMeleon lands where CheMeleon fine-tuning does (0.696, see
pxr_chemprop_chemeleon_phase1.py): the CheMeleon representation caps this analog
series, and stacking chemprop's prediction only recovers chemprop. The article's
gain came over a weak CheMeleon baseline; our from-scratch chemprop is already past it.
So TabICL is NOT a Workbench model framework.

Runs locally (not via ml_pipeline_launcher) against the shared FeatureSet
(openadmet_pxr_f1): fit on the `train` rows, score on the 253 `phase1_test` rows.
Every arm is scored against `pxr-reg-chemprop-phase1` on the same rows.

Arms:
  0  chemprop   pxr-reg-chemprop-phase1's `pxr_phase1_test` capture (the bar)
  1  tabicl     CheMeleon PCA -> TabICL
  2  tabicl_oof arm 1 + chemprop's prediction     (stacked: OOF on train, capture on test)

Arms 1-2 are averaged over SEEDS before the paired bootstrap, so the comparison
carries row noise only; the per-seed spread is reported alongside.

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

fs_name = "openadmet_pxr_f1"
baseline_model = "pxr-reg-chemprop-phase1"
id_col, target = "molecule_name", "pec50"
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


# Data: train rows fit, phase1_test rows score
df = FeatureSet(fs_name).pull_dataframe()[[id_col, "smiles", target, "split"]]
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "phase1_test"].reset_index(drop=True)
print(f"{len(train)} train / {len(test)} phase1_test rows")

# CheMeleon embedding -> standardize -> PCA, both fit on train only
emb_train, emb_test = chemeleon_embed(train["smiles"].tolist()), chemeleon_embed(test["smiles"].tolist())
scaler = StandardScaler().fit(emb_train)
pca = PCA(n_components=PCA_DIMS, random_state=0).fit(scaler.transform(emb_train))
X_train, X_test = pca.transform(scaler.transform(emb_train)), pca.transform(scaler.transform(emb_test))
print(f"PCA {emb_train.shape[1]} -> {PCA_DIMS}: {pca.explained_variance_ratio_.sum():.1%} variance kept")

# Chemprop's predictions: OOF on train (leak-free stacking feature), held-out capture on test
chemprop = Model(baseline_model)
oof = chemprop.get_inference_predictions("model_training").set_index(id_col)["prediction"]
held = chemprop.get_inference_predictions("pxr_phase1_test").set_index(id_col)["prediction"]
cp_train, cp_test = train[id_col].map(oof).to_numpy(), test[id_col].map(held).to_numpy()
assert not np.isnan(cp_train).any() and not np.isnan(cp_test).any(), "chemprop predictions missing rows"
X_train_oof, X_test_oof = np.column_stack([X_train, cp_train]), np.column_stack([X_test, cp_test])


def tabicl(seed: int, X_fit: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """CPU: on MPS a single 4k-row fit asks for ~8 GiB of shared GPU memory."""
    return TabICLRegressor(random_state=seed, device="cpu").fit(X_fit, train[target]).predict(X_pred)


arms = {
    "tabicl": lambda seed: tabicl(seed, X_train, X_test),
    "tabicl_oof": lambda seed: tabicl(seed, X_train_oof, X_test_oof),
}


def frame(pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({target: test[target].to_numpy(), "prediction": pred}, index=test[id_col])


base = frame(cp_test)
rows = [{"arm": "chemprop", "rae": rae(base), "seed_sd": np.nan, "delta": 0.0, "ci": "", "p_better": np.nan}]
for name, fit_predict in arms.items():
    per_seed = np.array([fit_predict(seed) for seed in SEEDS])
    seed_raes = [rae(frame(p)) for p in per_seed]
    cmp = bootstrap_compare(frame(per_seed.mean(axis=0)), base, rae)
    rows.append(
        {
            "arm": name,
            "rae": cmp["value_a"],
            "seed_sd": float(np.std(seed_raes)),
            "delta": cmp["delta"],
            "ci": f"[{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]",
            "p_better": cmp["p_a_better"],
        }
    )
    print(f"{name}: RAE {cmp['value_a']:.3f} (delta vs chemprop {cmp['delta']:+.3f})")

results = pd.DataFrame(rows)
print("\nphase1_test RAE (lower is better); delta/ci/p_better are paired vs chemprop")
print(results.to_string(index=False, float_format="%.3f"))
