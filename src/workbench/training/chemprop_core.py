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

import torch
from chemprop import models, nn


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
