"""Atom-JEPA serving: the fine-tuned fold model, its readout, and conformers from SMILES.

A fold model is the pretrained Atom-JEPA encoder plus a readout on its per-atom irreps,
saved in the released checkpoint layout so AtomJEPA.from_pretrained loads it offline.
Prediction averages over each molecule's conformers in target units.
:mod:`workbench.training.atomjepa_core` fine-tunes the fold models.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch
from atom_jepa import AtomJEPA
from atom_jepa.conformers import smiles_to_samples
from atom_jepa.data.collate import GraphCollator, move_batch
from atom_jepa.models.jepa_equiformer import pool_nodes
from safetensors.torch import save_file
from torch import nn

PREDICT_BATCH_SIZE = 64


# -----------------------------------------------------------------------------
# Conformers
# -----------------------------------------------------------------------------
def _conformers(smiles: str, n: int) -> list | None:
    """n seeded ETKDGv3 + MMFF94 conformers, or None when RDKit cannot build them."""
    try:
        return smiles_to_samples(smiles, num_conformers=n)
    except (ValueError, RuntimeError):
        return None


def build_conformers(smiles: list[str], n: int) -> list:
    """Per molecule, a list of n conformer samples (None where RDKit cannot build them).

    Runs on threads: RDKit releases the GIL while embedding, and the inference server calls
    this from a worker thread of a CUDA process, where forking is unsafe.
    """
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        return list(pool.map(_conformers, smiles, [n] * len(smiles)))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class Readout(nn.Module):
    """Mean-pooled l=0 features plus layer-normed, mean-pooled per-atom l=1/l=2 norms, into an MLP.

    Extra per-molecule features add the output of a zero-initialized MLP, so training starts
    from the structure-only prediction.
    """

    def __init__(self, channels: int, n_extra: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.norm = nn.LayerNorm(2 * channels)
        self.mlp = nn.Sequential(
            nn.Linear(3 * channels, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.extra = None
        if n_extra:
            self.extra = nn.Sequential(nn.Linear(n_extra, 64), nn.SiLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.extra[-1].weight)
            nn.init.zeros_(self.extra[-1].bias)

    def forward(self, atoms: torch.Tensor, batch: dict) -> torch.Tensor:
        """atoms: [n_atoms, 9, C], the l=0, l=1 (3) and l=2 (5) components per atom."""
        gidx, n = batch["node_graph_index"], batch["num_graphs"]
        norms = torch.cat([atoms[:, 1:4].pow(2).sum(1), atoms[:, 4:9].pow(2).sum(1)], dim=1).add(1e-8).sqrt()
        h = torch.cat([pool_nodes(atoms[:, 0], gidx, n), self.norm(pool_nodes(norms, gidx, n))], dim=1)
        y = self.mlp(h).squeeze(-1)
        if self.extra is not None:
            y = y + self.extra(batch["extra"]).squeeze(-1)
        return y


class FineTuned(nn.Module):
    """Atom-JEPA encoder + readout, predicting standardized targets."""

    def __init__(self, encoder: AtomJEPA, n_extra: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.readout = Readout(encoder.embedding_dim, n_extra, dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # the pretrained encoder's drop path stays off
        return self

    def forward(self, batch: dict, train_encoder: bool = True) -> torch.Tensor:
        with torch.set_grad_enabled(train_encoder and torch.is_grad_enabled()):
            atoms = self.encoder(batch, degrees="all", per_atom=True)
        return self.readout(atoms, batch)


@dataclass
class FoldModel:
    """A fine-tuned fold model with the scalings fit on its training rows."""

    model: FineTuned
    target_mean: float
    target_std: float
    extra_mean: np.ndarray | None = None
    extra_std: np.ndarray | None = None

    def scale_extra(self, extra: np.ndarray | None) -> np.ndarray | None:
        """Standardize extra features; a missing value becomes the training mean (0)."""
        if extra is None:
            return None
        return np.nan_to_num((extra - self.extra_mean) / self.extra_std).astype(np.float32)

    @torch.inference_mode()
    def predict(self, conformers: list, extra: np.ndarray | None = None) -> np.ndarray:
        """Per molecule, the mean prediction over its conformers, in target units."""
        self.model.eval()
        device = next(self.model.parameters()).device
        collator = graph_collator(self.model.encoder)
        extra = self.scale_extra(extra)
        pairs = [(i, conf) for i, confs in enumerate(conformers) for conf in confs]
        preds = np.empty(len(pairs))
        for start in range(0, len(pairs), PREDICT_BATCH_SIZE):
            chunk = pairs[start : start + PREDICT_BATCH_SIZE]
            batch = move_batch(collator([sample(conf, extra=extra, i=i) for i, conf in chunk]), device)
            preds[start : start + len(chunk)] = self.model(batch).float().cpu().numpy()
        mol = np.array([i for i, _ in pairs])
        native = preds * self.target_std + self.target_mean
        return np.bincount(mol, weights=native, minlength=len(conformers)) / np.bincount(mol, minlength=len(conformers))

    def save(self, path: str):
        """The encoder in the released checkpoint layout (AtomJEPA.from_pretrained loads it), plus the readout."""
        os.makedirs(path, exist_ok=True)
        encoder = self.model.encoder
        # Blocks share their constant rotation buffers; safetensors needs each key as its own
        # contiguous tensor
        state = {
            k: v.detach().clone(memory_format=torch.contiguous_format) for k, v in encoder.encoder.state_dict().items()
        }
        save_file(state, f"{path}/model.safetensors")
        with open(f"{path}/config.json", "w", encoding="utf-8") as f:
            json.dump({"eqv3_cfg": asdict(encoder.config)}, f)
        readout = {
            "state": self.model.readout.state_dict(),
            "n_extra": 0 if self.extra_mean is None else len(self.extra_mean),
            "dropout": self.model.readout.dropout,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "extra_mean": None if self.extra_mean is None else torch.from_numpy(self.extra_mean),
            "extra_std": None if self.extra_std is None else torch.from_numpy(self.extra_std),
        }
        torch.save(readout, f"{path}/readout.pt")

    @classmethod
    def load(cls, path: str, device=None) -> "FoldModel":
        encoder = AtomJEPA.from_pretrained(path, device=device, verbose=False)
        readout = torch.load(f"{path}/readout.pt", map_location=encoder.device)
        model = FineTuned(encoder, readout["n_extra"], readout["dropout"]).to(encoder.device)
        model.readout.load_state_dict(readout["state"])
        extra_mean, extra_std = readout["extra_mean"], readout["extra_std"]
        return cls(
            model.eval(),
            readout["target_mean"],
            readout["target_std"],
            None if extra_mean is None else extra_mean.numpy(),
            None if extra_std is None else extra_std.numpy(),
        )


def graph_collator(encoder: AtomJEPA) -> GraphCollator:
    """Batches samples into one radius graph at the encoder's cutoff."""
    return GraphCollator(cutoff=encoder.config.max_radius, max_num_elements=encoder.config.max_num_elements)


def sample(conf: dict, *, extra: np.ndarray | None, i: int, y: np.ndarray | None = None) -> dict:
    """A conformer plus row i's graph-level tensors (the collator stacks them)."""
    out = dict(conf)
    if extra is not None:
        out["extra"] = torch.from_numpy(extra[i])
    if y is not None:
        out["y"] = torch.tensor(y[i], dtype=torch.float32)
    return out
