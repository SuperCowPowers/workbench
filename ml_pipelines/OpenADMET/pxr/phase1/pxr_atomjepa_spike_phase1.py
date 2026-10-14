"""PXR phase-1 spike: TabICL on frozen Atom-JEPA embeddings (+ primary-screen readouts).

Atom-JEPA (code MIT, weights CC BY 4.0, https://github.com/khelverskovp/atom-jepa,
arXiv:2610.08400) is an EquiformerV3 encoder (39M params) pretrained by latent prediction
on ~19M Uni-Mol molecules. Its frozen encoder turns a SMILES (through one seeded ETKDGv3 +
MMFF94 conformer, the paper's protocol) into a 256-d embedding per block. The paper's ADMET
wins are fine-tuned; its frozen-encoder MLP probes trailed Morgan + RDKit features on Biogen.
This spike asks whether the frozen embedding earns a feature endpoint, as Monroe's did.

Runs locally (not via ml_pipeline_launcher) against openadmet_pxr_readout_monroe: fit on
the `train` rows, score on the 253 `phase1_test` rows. Every arm gets EMBEDDING_FEATURES
embedding columns plus (where named) the two readouts:

  monroe_readout           the highest-variance Monroe columns (variance on train rows),
                           the Monroe ablation's best arm, refit here for a head to head
  atomjepa                 the highest-variance columns of the final (normed) block
  atomjepa_readout         the same, plus the readouts
  atomjepa_blocks_readout  an equal share of the highest-variance columns from each of the
                           8 blocks (block scales differ, so a global top-k favors one
                           block), plus the readouts

TabICL arms are averaged over SEEDS before the paired bootstrap, so the comparison carries
row noise only. PAIRS are compared head to head on the same 253 compounds.

Results (phase1_test, 3 seeds, paired RAE delta with 95% CI):

    arm                      RAE    MAE    vs chemprop               vs chemprop_readout
    chemprop                 0.591  0.472
    chemprop_readout         0.569  0.454
    monroe_readout           0.537  0.429  -0.054 [-0.092, -0.016]   -0.032 [-0.064, -0.004]
    atomjepa                 0.672  0.536  +0.081 [+0.025, +0.140]   +0.103 [+0.051, +0.153]
    atomjepa_readout         0.555  0.444  -0.035 [-0.078, +0.011]   -0.014 [-0.043, +0.017]
    atomjepa_blocks_readout  0.558  0.445  -0.033 [-0.076, +0.013]   -0.011 [-0.039, +0.018]

Head to head (paired RAE, first minus second):

    atomjepa_readout         vs monroe_readout     +0.019 [-0.015, +0.053]
    atomjepa_blocks_readout  vs monroe_readout     +0.021 [-0.010, +0.055]
    atomjepa_blocks_readout  vs atomjepa_readout   +0.002 [-0.016, +0.021]

Frozen Atom-JEPA alone trails chemprop, and frozen Monroe in the same position (0.616,
pxr_monroe_spike_phase1.py). With the readouts it trails Monroe + readouts (Monroe wins
86% of resamples), and the per-block columns add nothing. The frozen embedding does not
earn a feature endpoint; the paper's gains come from fine-tuning.

Encodes on MPS when available (~6 molecules/s on an M1 Max, ~6 GB of GPU memory):

    uv run --with "atom-jepa[rdkit]==0.3.0" python pxr_atomjepa_spike_phase1.py
"""

import numpy as np
import pandas as pd
import torch
from atom_jepa import AtomJEPA
from atom_jepa.conformers import smiles_to_samples

from workbench.api import FeatureSet, Model
from workbench.endpoints.inference import decompress_features
from workbench.endpoints.tabicl_utils import predict_with_std
from workbench.training.tabicl_core import fit_tabicl
from workbench.utils.metrics_utils import bootstrap_compare

fs_name = "openadmet_pxr_readout_monroe"
id_col, target = "molecule_name", "pec50"
READOUTS = ["lfc_8um_readout", "lfc_33um_readout"]
BASELINES = {"chemprop": "pxr-reg-chemprop-phase1", "chemprop_readout": "pxr-reg-chemprop-readout-phase1"}
EMBEDDING_FEATURES = 100
HYPERPARAMETERS = {"n_estimators": 8, "batch_size": 1, "pca_components": None}
SEEDS = [0, 1, 2]
DEVICE = "mps" if torch.backends.mps.is_available() else None  # None: CUDA if present, else CPU
EMBED_CHUNK = 256


def rae(frame: pd.DataFrame) -> float:
    """Relative absolute error against the mean predictor (the challenge's headline)."""
    y = frame[target]
    return float((frame["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


def conformer(smiles: str) -> dict | None:
    """One seeded ETKDGv3 + MMFF94 conformer, or None when RDKit cannot build one."""
    try:
        return smiles_to_samples(smiles)[0]
    except (ValueError, RuntimeError):
        return None


# Data: the Monroe FeatureSet's rows, so both embeddings cover the same molecules
df = FeatureSet(fs_name).pull_dataframe()[[id_col, "smiles", "monroe", target, "split"] + READOUTS]
df, monroe_cols = decompress_features(df, ["monroe"], ["monroe"])

samples = df["smiles"].map(conformer)
embedded = samples.notna()
assert embedded[df["split"] == "phase1_test"].all(), "no conformer for a phase1_test molecule"
df = df[embedded].reset_index(drop=True)
print(f"Atom-JEPA conformers for {embedded.sum()} of {len(embedded)} molecules", flush=True)

# The final normed block, then blocks 1..8 -> [molecules, 9, 256]. Per-edge activations
# dominate memory, so batches stay small, and the MPS cache (which grows with every new
# batch shape) is emptied between chunks.
encoder = AtomJEPA.from_pretrained("molecules", device=DEVICE, precision="fp32")
blocks = list(range(1, encoder.num_layers + 1))
structures = samples[embedded].tolist()
chunks = []
for start in range(0, len(structures), EMBED_CHUNK):
    chunk = structures[start : start + EMBED_CHUNK]
    chunks.append(encoder.embed(chunk, layers=["last", *blocks], batch_size=8).numpy())
    if DEVICE == "mps":
        torch.mps.empty_cache()
    print(f"embedded {start + len(chunk)} of {len(structures)}", flush=True)
emb = np.concatenate(chunks)
last_cols = [f"ajepa_{i}" for i in range(emb.shape[2])]
block_cols = {b: [f"ajepa_b{b}_{i}" for i in range(emb.shape[2])] for b in blocks}
df = pd.concat(
    [df, pd.DataFrame(emb[:, 0], columns=last_cols)] + [pd.DataFrame(emb[:, b], columns=block_cols[b]) for b in blocks],
    axis=1,
)

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "phase1_test"].reset_index(drop=True)
print(f"{len(train)} train / {len(test)} phase1_test rows", flush=True)


def top_variance(columns: list[str], k: int) -> list[str]:
    """The k highest-variance columns, variance on train rows."""
    return list(train[columns].var().nlargest(k).index)


def arm(columns: list[str], readouts: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = columns + (READOUTS if readouts else [])
    return train[keep], test[keep]


def frame(pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({target: test[target].to_numpy(), "prediction": pred}, index=test[id_col])


def captured(model_name: str) -> pd.DataFrame:
    """A deployed model's held-out `pxr_phase1_test` predictions, aligned to `test`."""
    held = Model(model_name).get_inference_predictions("pxr_phase1_test").set_index(id_col)["prediction"]
    pred = test[id_col].map(held).to_numpy()
    assert not np.isnan(pred).any(), f"{model_name} capture is missing phase1_test rows"
    return frame(pred)


per_block = EMBEDDING_FEATURES // len(blocks)
block_top = [c for b in blocks for c in top_variance(block_cols[b], per_block)]
ARMS = {
    "monroe_readout": arm(top_variance(monroe_cols, EMBEDDING_FEATURES), readouts=True),
    "atomjepa": arm(top_variance(last_cols, EMBEDDING_FEATURES), readouts=False),
    "atomjepa_readout": arm(top_variance(last_cols, EMBEDDING_FEATURES), readouts=True),
    "atomjepa_blocks_readout": arm(block_top, readouts=True),
}

# Head-to-head comparisons between arms (first minus second; negative = first is better)
PAIRS = [
    ("atomjepa_readout", "monroe_readout"),
    ("atomjepa_blocks_readout", "monroe_readout"),
    ("atomjepa_blocks_readout", "atomjepa_readout"),
]

baselines = {name: captured(model_name) for name, model_name in BASELINES.items()}
rows = [{"arm": name, "rae": rae(f), "mae": mae(f), "seeds": "deployed"} for name, f in baselines.items()]

averaged = {}  # arm -> seed-averaged prediction frame, for the head-to-head comparisons
for name, (X_train, X_test) in ARMS.items():
    per_seed = []
    for seed in SEEDS:
        model, _ = fit_tabicl({**HYPERPARAMETERS, "seed": seed}, X_train, train[target], cache=False)
        per_seed.append(predict_with_std(model, X_test)[0])
    averaged[name] = frame(np.mean(per_seed, axis=0))
    single = [rae(frame(p)) for p in per_seed]
    row = {
        "arm": name,
        "rae": rae(averaged[name]),
        "mae": mae(averaged[name]),
        "seeds": f"{len(SEEDS)} averaged",
        "single-seed rae": f"{np.mean(single):.3f} ± {np.std(single):.3f}",
    }
    for base, base_frame in baselines.items():
        cmp = bootstrap_compare(averaged[name], base_frame, rae)
        row[f"vs {base}"] = f"{cmp['delta']:+.3f} [{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]"
    rows.append(row)
    summary = f"RAE {row['rae']:.3f} averaged, {row['single-seed rae']} single seed"
    print(f"{name} ({X_train.shape[1]} features): {summary}", flush=True)

print(f"\nphase1_test, {len(test)} compounds (RAE: lower is better; deltas are paired RAE, 95% bootstrap)")
print(pd.DataFrame(rows).fillna("").to_string(index=False, float_format="%.3f"))

head_to_head = []
for a, b in PAIRS:
    cmp = bootstrap_compare(averaged[a], averaged[b], rae)
    head_to_head.append(
        {
            "arm": a,
            "vs": b,
            "delta": f"{cmp['delta']:+.3f}",
            "95% CI": f"[{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]",
            "p(first better)": f"{cmp['p_a_better']:.2f}",
        }
    )
print("\nhead to head, paired RAE (first minus second; negative = first is better)")
print(pd.DataFrame(head_to_head).to_string(index=False))
