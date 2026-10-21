"""Atom-JEPA fine-tuning for single-target regression: the fold trainer behind the atomjepa template.

Ports the ADMET recipe from github.com/khelverskovp/atom-jepa (finetuning/admet): a readout
on the encoder's per-atom irreps, Huber loss on standardized targets, AdamW with separate
encoder and head learning rates, the encoder frozen for the first epochs, warmup + cosine
stepped per epoch, and an EMA of the whole model that every validation, checkpoint pick and
prediction uses. Each molecule carries several conformers: training draws one at random per
epoch, and prediction averages over all of them in target units.
"""

import math

import numpy as np
import torch
from atom_jepa import AtomJEPA
from atom_jepa.data.collate import move_batch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from workbench.endpoints.atomjepa_utils import FineTuned, FoldModel, graph_collator, sample
from workbench.training.foundation_models import resolve_foundation_checkpoint


def pretrained_encoder_dir() -> str:
    """The released molecules encoder, from the foundation registry, as the folder from_pretrained loads."""
    resolve_foundation_checkpoint("atomjepa_config")
    return str(resolve_foundation_checkpoint("atomjepa").parent)


def default_device() -> torch.device:
    """CUDA when present, else CPU: MPS trains this model far slower than CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    Each epoch's validation MAE averages over the first `selection_conformers` conformers per
    molecule; the returned validation predictions average over all of them.

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

    encoder = AtomJEPA.from_pretrained(pretrained_encoder_dir(), device=device, verbose=seed == 0)
    model = FineTuned(encoder, 0 if train_extra is None else train_extra.shape[1], hp["dropout"]).to(device)
    # Recompute each encoder block's activations in the backward pass rather than storing them:
    # a batch of 32 drug-sized molecules otherwise outgrows a 24 GB GPU
    model.encoder.set_grad_checkpointing(True)
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

    selection_conformers = [confs[: hp["selection_conformers"]] for confs in val_conformers]
    best_mae, best_state = math.inf, None
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

        val_mae = float(np.abs(ema_fold.predict(selection_conformers, val_extra) - val_y).mean())
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
        print(
            f"  epoch {epoch + 1}/{hp['epochs']}  train_loss {np.mean(losses):.4f}  val_mae {val_mae:.4f}"
            f"{'  *' if improved else ''}{'  (encoder frozen)' if frozen else ''}",
            flush=True,
        )

    ema.module.load_state_dict(best_state)
    return ema_fold, ema_fold.predict(val_conformers, val_extra)
