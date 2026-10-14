"""Atom-JEPA fine-tuning for single-target regression: the fold trainer behind atomjepa.template.

Ports the ADMET recipe from github.com/khelverskovp/atom-jepa (finetuning/admet): a readout
on the encoder's per-atom irreps, Huber loss on standardized targets, AdamW with separate
encoder and head learning rates, the encoder frozen for the first epochs, warmup + cosine
stepped per epoch, and an EMA of the whole model that every validation, checkpoint pick and
prediction uses. Each molecule carries several conformers: training draws one at random per
epoch, and prediction averages over all of them in target units.
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch
from atom_jepa import AtomJEPA
from atom_jepa.conformers import smiles_to_samples
from atom_jepa.data.collate import GraphCollator, move_batch
from atom_jepa.models.jepa_equiformer import pool_nodes
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

PREDICT_BATCH_SIZE = 64


def default_device() -> torch.device:
    """CUDA when present, else CPU: MPS trains this model far slower than CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Conformers
# -----------------------------------------------------------------------------
def _conformers(smiles: str, n: int) -> list | None:
    """n seeded ETKDGv3 + MMFF94 conformers as (atomic_numbers, positions) arrays, or None.

    Arrays rather than tensors: tensors returned from worker processes travel as shared
    memory handles, one file descriptor each.
    """
    try:
        samples = smiles_to_samples(smiles, num_conformers=n)
    except (ValueError, RuntimeError):
        return None
    return [(s["atomic_numbers"].numpy(), s["node_coordinates"].numpy()) for s in samples]


def _as_samples(confs: list | None) -> list | None:
    if confs is None:
        return None
    return [{"atomic_numbers": torch.from_numpy(z), "node_coordinates": torch.from_numpy(pos)} for z, pos in confs]


def build_conformers(smiles: list[str], n: int) -> list:
    """Per molecule, a list of n conformer samples (None where RDKit cannot build them)."""
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        return [_as_samples(c) for c in pool.map(_conformers, smiles, [n] * len(smiles), chunksize=16)]


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
        with open(f"{path}/config.json", "w") as f:
            json.dump({"eqv3_cfg": asdict(encoder.config)}, f)
        readout = {
            "state": self.model.readout.state_dict(),
            "n_extra": 0 if self.extra_mean is None else len(self.extra_mean),
            "dropout": self.model.readout.mlp[2].p,
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


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def lr_scale(epoch: int, hp: dict, base_lr: float, encoder: bool) -> float:
    """Linear warmup then cosine to min_lr; the encoder re-warms over the epochs after unfreezing."""
    warmup, epochs, freeze = hp["warmup_epochs"], hp["epochs"], hp["freeze_encoder_epochs"]
    if epoch < warmup:
        scale = (epoch + 1) / warmup
    else:
        floor = hp["min_lr"] / base_lr
        scale = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    if encoder and freeze and freeze <= epoch < freeze + warmup:
        scale *= (epoch - freeze + 1) / warmup
    return scale


def train_fold(
    hp: dict,
    train_conformers: list,
    train_y: np.ndarray,
    val_conformers: list,
    val_y: np.ndarray,
    train_extra: np.ndarray | None = None,
    val_extra: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[FoldModel, np.ndarray]:
    """Fine-tune one fold model; the EMA weights from the epoch with the lowest validation MAE win.

    Returns the fold model and its predictions for the validation rows (target units).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = default_device()

    extra_mean = extra_std = None
    if train_extra is not None:
        extra_mean = np.nanmean(train_extra, axis=0)
        extra_std = np.where(np.nanstd(train_extra, axis=0) > 0, np.nanstd(train_extra, axis=0), 1.0)
    target_mean, target_std = float(train_y.mean()), float(train_y.std())

    encoder = AtomJEPA.from_pretrained("molecules", device=device, verbose=seed == 0)
    model = FineTuned(encoder, 0 if train_extra is None else train_extra.shape[1], hp["dropout"]).to(device)
    fold = FoldModel(model, target_mean, target_std, extra_mean, extra_std)
    y_scaled = ((train_y - target_mean) / target_std).astype(np.float32)
    x_scaled = fold.scale_extra(train_extra)
    collator = graph_collator(encoder)

    base_lrs = [hp["encoder_lr"], hp["head_lr"]]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": hp["encoder_lr"], "weight_decay": 0.0},
            {"params": model.readout.parameters(), "lr": hp["head_lr"], "weight_decay": hp["head_weight_decay"]},
        ],
        fused=device.type == "cuda",
    )
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(hp["ema_decay"]))
    ema_fold = FoldModel(ema.module, target_mean, target_std, extra_mean, extra_std)

    best_mae, best_state, best_pred = math.inf, None, None
    for epoch in range(hp["epochs"]):
        frozen = epoch < hp["freeze_encoder_epochs"]
        for group, base_lr, is_encoder in zip(optimizer.param_groups, base_lrs, (True, False)):
            group["lr"] = base_lr * lr_scale(epoch, hp, base_lr, is_encoder)

        model.train()
        order, losses = rng.permutation(len(train_conformers)), []
        for start in range(0, len(order), hp["batch_size"]):
            rows = order[start : start + hp["batch_size"]]
            confs = [train_conformers[i][rng.integers(len(train_conformers[i]))] for i in rows]
            batch = move_batch(
                collator([sample(c, extra=x_scaled, i=i, y=y_scaled) for c, i in zip(confs, rows)]), device
            )
            loss = F.smooth_l1_loss(model(batch, train_encoder=not frozen), batch["y"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
            ema.update_parameters(model)
            losses.append(loss.item())

        val_pred = ema_fold.predict(val_conformers, val_extra)
        val_mae = float(np.abs(val_pred - val_y).mean())
        improved = val_mae < best_mae
        if improved:
            best_mae, best_pred = val_mae, val_pred
            best_state = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
        print(
            f"  epoch {epoch + 1}/{hp['epochs']}  train_loss {np.mean(losses):.4f}  val_mae {val_mae:.4f}"
            f"{'  *' if improved else ''}{'  (encoder frozen)' if frozen else ''}",
            flush=True,
        )

    ema.module.load_state_dict(best_state)
    return ema_fold, best_pred
