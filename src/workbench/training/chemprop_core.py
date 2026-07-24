"""Shared chemprop model construction — the config → MPNN mapping.

Training-only (per the :mod:`workbench.training` contract); imported **only inside
a template's ``__main__``** (deferred). Both the chemprop template's training path
and the HPO objective (:mod:`workbench.training.chemprop_hpo`) build models through
:func:`build_mpnn_model`, so a searched config maps to the *same* architecture and
optimizer schedule it will get when the winner is published — that shared builder
is the HPO parity guarantee.

Requires chemprop (present in both the training and inference images); it is never
imported at endpoint load, so the training-only classification is about the
:mod:`workbench.training` deferred-import rule, not chemprop availability.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from chemprop import data, models, nn
from lightning import pytorch as pl

from workbench.endpoints.chemprop_utils import create_molecule_datapoints, safe_batch_size


def load_foundation_weights(from_foundation: str) -> tuple:
    """Load pretrained MPNN weights from a foundation model.

    Args:
        from_foundation: "CheMeleon" or a path to a .pt file.

    Returns:
        tuple: (message_passing, aggregation) modules.
    """
    import urllib.request
    from pathlib import Path

    print(f"Loading foundation model: {from_foundation}")

    if from_foundation.lower() == "chemeleon":
        # Download from Zenodo if not cached
        cache_dir = Path.home() / ".chemprop" / "foundation"
        cache_dir.mkdir(parents=True, exist_ok=True)
        chemeleon_path = cache_dir / "chemeleon_mp.pt"

        if not chemeleon_path.exists():
            print("  Downloading CheMeleon weights from Zenodo...")
            urllib.request.urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", chemeleon_path)
            print(f"  Downloaded to {chemeleon_path}")

        ckpt = torch.load(chemeleon_path, weights_only=True)
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded CheMeleon MPNN (hidden_dim={mp.output_dim})")
        return mp, nn.MeanAggregation()

    if not os.path.exists(from_foundation):
        raise ValueError(f"Foundation model not found: {from_foundation}. Use 'CheMeleon' or a valid .pt path.")

    ckpt = torch.load(from_foundation, weights_only=False)
    if "hyper_parameters" in ckpt and "state_dict" in ckpt:
        # CheMeleon-style checkpoint
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded custom foundation weights (hidden_dim={mp.output_dim})")
        return mp, nn.MeanAggregation()

    # Full MPNN model file
    pretrained = models.MPNN.load_from_file(from_foundation)
    print(f"  Loaded custom MPNN (hidden_dim={pretrained.message_passing.output_dim})")
    return pretrained.message_passing, pretrained.agg


def build_ffn(
    task: str,
    input_dim: int,
    hyperparameters: dict,
    num_classes: int | None,
    n_targets: int,
    output_transform,
    task_weights,
    use_bounded_loss: bool = False,
):
    """Build the task-specific FFN head.

    When ``use_bounded_loss`` is True, swaps the regression criterion for its bounded
    variant (BoundedMAE/BoundedMSE) so gt_mask/lt_mask censoring is honored at train
    time. The mask data itself flows through MoleculeDatapoint, not here.
    """
    dropout = hyperparameters["dropout"]
    ffn_hidden_dim = hyperparameters["ffn_hidden_dim"]
    ffn_num_layers = hyperparameters["ffn_num_layers"]

    if task == "classification" and num_classes is not None:
        return nn.MulticlassClassificationFFN(
            n_classes=num_classes,
            input_dim=input_dim,
            hidden_dim=ffn_hidden_dim,
            n_layers=ffn_num_layers,
            dropout=dropout,
        )

    from chemprop.nn.metrics import MAE, MSE, BoundedMAE, BoundedMSE

    if use_bounded_loss:
        criterion_map = {"mae": BoundedMAE, "mse": BoundedMSE}
    else:
        criterion_map = {"mae": MAE, "mse": MSE}
    criterion_name = hyperparameters.get("criterion", "mae")
    if criterion_name not in criterion_map:
        raise ValueError(f"Unknown criterion '{criterion_name}'. Supported: {list(criterion_map.keys())}")

    criterion_kwargs = {"task_weights": task_weights} if task_weights is not None else {}
    return nn.RegressionFFN(
        input_dim=input_dim,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
        dropout=dropout,
        n_tasks=n_targets,
        output_transform=output_transform,
        criterion=criterion_map[criterion_name](**criterion_kwargs),
    )


def build_mpnn_model(
    hyperparameters: dict,
    task: str = "regression",
    num_classes: int | None = None,
    n_targets: int = 1,
    n_extra_descriptors: int = 0,
    x_d_transform=None,
    output_transform=None,
    task_weights=None,
    use_bounded_loss: bool = False,
):
    """Build an MPNN model from a hyperparameters dict, optionally loading pretrained weights."""
    from_foundation = hyperparameters.get("from_foundation")

    if from_foundation:
        mp, agg = load_foundation_weights(from_foundation)
        ffn_input_dim = mp.output_dim + n_extra_descriptors
    else:
        mp = nn.BondMessagePassing(
            d_h=hyperparameters["hidden_dim"],
            depth=hyperparameters["depth"],
            dropout=hyperparameters["dropout"],
        )
        agg = nn.NormAggregation()
        ffn_input_dim = hyperparameters["hidden_dim"] + n_extra_descriptors

    ffn = build_ffn(
        task,
        ffn_input_dim,
        hyperparameters,
        num_classes,
        n_targets,
        output_transform,
        task_weights,
        use_bounded_loss=use_bounded_loss,
    )
    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        metrics=None,
        X_d_transform=x_d_transform,
        warmup_epochs=hyperparameters["warmup_epochs"],
        init_lr=hyperparameters["init_lr"],
        max_lr=hyperparameters["max_lr"],
        final_lr=hyperparameters["final_lr"],
    )


@dataclass
class FoldSpec:
    """The parts of a chemprop training run that are invariant across folds.

    Built once from the resolved hyperparameters, then passed to every
    :func:`train_chemprop_fold` call in the ensemble.
    """

    hyperparameters: dict
    task: str = "regression"
    model_type: str = "regressor"
    smiles_column: str = "smiles"
    n_targets: int = 1
    num_classes: int | None = None
    n_extra_descriptors: int = 0
    task_weights: Any = None
    use_bounded_loss: bool = False
    use_extra_features: bool = False
    enable_progress_bar: bool = True
    verbose: bool = True
    # Dataloader workers; defaults to the CPU count capped at 8. Concurrent trials must
    # divide this down — every worker is a process, and oversubscription starves the GPU.
    num_workers: int = field(default_factory=lambda: min(os.cpu_count() or 4, 8))


def train_chemprop_fold(
    spec: FoldSpec,
    train_df,
    val_df,
    train_targets,
    val_targets,
    *,
    fold_idx: int = 0,
    checkpoint_dir: str | None = None,
    train_extra=None,
    val_extra=None,
    val_extra_raw=None,
    train_gt=None,
    train_lt=None,
    val_gt=None,
    val_lt=None,
):
    """Train one ensemble member and predict its validation fold.

    The single definition of "train one chemprop model" — used both to publish an
    ensemble and to score an HPO trial, so a searched config is evaluated exactly as it
    will be deployed. Handles extra descriptors, bounded-loss censoring masks, and
    two-phase foundation fine-tuning.

    Args:
        spec: the fold-invariant :class:`FoldSpec`.
        train_df / val_df: this fold's frames (must carry ``spec.smiles_column``).
        train_targets / val_targets: aligned target arrays.
        fold_idx: ensemble member index — offsets the seed so members differ.
        checkpoint_dir: when given, the best-``val_loss`` checkpoint is written here and
            reloaded before predicting. Without it the final-epoch weights are used, which
            scores a *different* model than the template publishes — pass a temp dir
            rather than omitting it.
        train_extra / val_extra: scaled extra descriptors; ``val_extra_raw`` is the
            unscaled copy used for prediction (the model's ``x_d_transform`` rescales).
        train_gt / train_lt / val_gt / val_lt: bounded-loss censoring masks.

    Returns:
        tuple: ``(mpnn, val_predictions)`` — the fitted model in eval mode and its
        predictions on ``val_df`` in original target units.
    """
    hp = spec.hyperparameters
    batch_size = hp["batch_size"]

    train_dps, _ = create_molecule_datapoints(
        train_df[spec.smiles_column].tolist(), train_targets, train_extra, gt_mask=train_gt, lt_mask=train_lt
    )
    val_dps, _ = create_molecule_datapoints(
        val_df[spec.smiles_column].tolist(), val_targets, val_extra, gt_mask=val_gt, lt_mask=val_lt
    )
    train_dataset, val_dataset = data.MoleculeDataset(train_dps), data.MoleculeDataset(val_dps)

    x_d_transform = None
    if spec.use_extra_features:
        scaler = train_dataset.normalize_inputs("X_d")
        val_dataset.normalize_inputs("X_d", scaler)
        x_d_transform = nn.ScaleTransform.from_standard_scaler(scaler)

    output_transform = None
    if spec.model_type in ["regressor", "uq_regressor"]:
        target_scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(target_scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)

    # Cache featurized MolGraphs so dataloader workers don't re-featurize every epoch (CPU
    # bottleneck). Setting cache=True eagerly precomputes all MolGraphs.
    _cache_t0 = time.time()
    train_dataset.cache = True
    val_dataset.cache = True
    if spec.verbose:
        print(
            f"[cache] featurization cache ENABLED — {len(train_dataset)} train + {len(val_dataset)} val "
            f"MolGraphs precomputed in {time.time() - _cache_t0:.1f}s",
            flush=True,
        )

    nw = spec.num_workers
    # persistent_workers / prefetch_factor are only valid with worker processes; PyTorch
    # rejects them when num_workers == 0.
    loader_kwargs = {"num_workers": nw, "pin_memory": True}
    if nw > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = data.build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = data.build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    pl.seed_everything(hp["seed"] + fold_idx)
    mpnn = build_mpnn_model(
        hp,
        task=spec.task,
        num_classes=spec.num_classes,
        n_targets=spec.n_targets,
        n_extra_descriptors=spec.n_extra_descriptors,
        x_d_transform=x_d_transform,
        output_transform=output_transform,
        task_weights=spec.task_weights,
        use_bounded_loss=spec.use_bounded_loss,
    )

    def _set_mpnn_frozen(frozen: bool):
        for param in mpnn.message_passing.parameters():
            param.requires_grad = not frozen
        for param in mpnn.agg.parameters():
            param.requires_grad = not frozen

    def _make_trainer(max_epochs: int, save_checkpoint: bool = False):
        callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", patience=hp["patience"], mode="min")]
        if save_checkpoint and checkpoint_dir:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=f"best_{fold_idx}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                )
            )
        # Only enable checkpointing on the phase that actually saves — otherwise Lightning
        # adds a default ModelCheckpoint writing throwaway files (e.g. foundation phase 1).
        save = save_checkpoint and bool(checkpoint_dir)
        return pl.Trainer(
            accelerator="auto",
            # One device per fold: on a multi-GPU box Lightning otherwise auto-selects DDP
            # (data parallelism) and deadlocks, since phase-2 publish is a single process.
            # DDP is the wrong tool anyway — these models are small and dataloader-bound, so
            # splitting one fold across GPUs adds sync overhead for no gain. The utilization
            # win is *task* parallelism (independent folds one-per-GPU); see the "Parallelize
            # the publish folds" backlog item in docs/planning/hpo_support.md.
            devices=1,
            max_epochs=max_epochs,
            precision="16-mixed",
            logger=False,
            enable_progress_bar=spec.enable_progress_bar,
            enable_checkpointing=save,
            callbacks=callbacks,
        )

    freeze_mpnn_epochs = hp.get("freeze_mpnn_epochs", 0)
    _fit_t0 = time.time()
    if hp.get("from_foundation") and freeze_mpnn_epochs > 0:
        # Phase 1: freeze the MPNN and train the FFN head only.
        if spec.verbose:
            print(f"Phase 1: Training with frozen MPNN for {freeze_mpnn_epochs} epochs...")
        _set_mpnn_frozen(True)
        _make_trainer(freeze_mpnn_epochs).fit(mpnn, train_loader, val_loader)

        # Phase 2: unfreeze and fine-tune everything.
        if spec.verbose:
            print("Phase 2: Unfreezing MPNN, continuing training...")
        _set_mpnn_frozen(False)
        trainer = _make_trainer(max(1, hp["max_epochs"] - freeze_mpnn_epochs), save_checkpoint=True)
        trainer.fit(mpnn, train_loader, val_loader)
    else:
        trainer = _make_trainer(hp["max_epochs"], save_checkpoint=True)
        trainer.fit(mpnn, train_loader, val_loader)
    if spec.verbose:
        print(f"[timing] fold {fold_idx + 1} fit: {time.time() - _fit_t0:.1f}s", flush=True)

    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False)
        mpnn.load_state_dict(checkpoint["state_dict"])
    mpnn.eval()

    _pred_t0 = time.time()
    val_preds = predict_chemprop_frame(mpnn, spec, val_df, targets=val_targets, extra=val_extra_raw)
    if spec.verbose:
        print(f"[timing] fold {fold_idx + 1} predict: {time.time() - _pred_t0:.1f}s", flush=True)
    return mpnn, val_preds


def predict_chemprop_frame(mpnn, spec: FoldSpec, df, targets=None, extra=None):
    """Predict a frame with a fitted chemprop model, in original target units.

    ``extra`` must be the *unscaled* descriptors — the model's ``x_d_transform`` applies
    its own scaling, so pre-scaled input would be double-scaled.

    Returns:
        np.ndarray: predictions shaped ``(n_rows, n_outputs)``.
    """
    dps, _ = create_molecule_datapoints(df[spec.smiles_column].tolist(), targets, extra)
    dataset = data.MoleculeDataset(dps)
    loader = data.build_dataloader(
        dataset,
        batch_size=safe_batch_size(len(dataset), spec.hyperparameters["batch_size"]),
        shuffle=False,
        num_workers=spec.num_workers,
        pin_memory=True,
    )
    # fp32 (the Trainer default), not the training fit's "16-mixed": mixed precision is a
    # training-throughput technique. All inference — OOF here, the manual std/calibration
    # loops, and the serving endpoint — is fp32, so UQ calibration is fit on the same numbers
    # production emits. Keep this fp32; don't "restore parity" with the fit's precision.
    trainer = pl.Trainer(
        accelerator="auto", devices=1, logger=False, enable_progress_bar=False, enable_checkpointing=False
    )
    mpnn.eval()
    with torch.inference_mode():
        preds = np.concatenate([p.numpy() for p in trainer.predict(mpnn, loader)], axis=0)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds.squeeze(axis=1)
    return preds
