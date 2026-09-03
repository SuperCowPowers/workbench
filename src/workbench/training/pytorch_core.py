"""Shared PyTorch tabular fold training — the config → MLP mapping.

Training-only (per the :mod:`workbench.training` contract); imported **only inside
the PyTorch template's ``__main__``** (deferred). Both the template's publish loop
and the HPO objective (:mod:`workbench.training.pytorch_hpo`) train ensemble members
through :func:`train_pytorch_fold`, so a searched config maps to the *same*
architecture, seed schedule, and optimizer it will get when the winner is published —
that shared recipe is the HPO parity guarantee.

The model/optimizer primitives (``create_model``/``train_model``/``prepare_data``)
live in :mod:`workbench.endpoints.pytorch_utils` because inference loads them; this
module owns the training *recipe* built on top of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PyTorchFoldSpec:
    """The parts of a PyTorch tabular training run that are invariant across folds.

    Built once from the resolved hyperparameters and the template's fitted
    preprocessing (scaler, category mappings), then passed to every
    :func:`train_pytorch_fold` call in the ensemble. The last three fields carry the
    fitted frame-alignment state :func:`align_frame` applies to raw frames.
    """

    hyperparameters: dict
    target: str
    continuous_cols: list
    categorical_cols: list = field(default_factory=list)
    category_mappings: dict = field(default_factory=dict)
    categorical_cardinalities: list = field(default_factory=list)
    scaler: Any = None
    n_outputs: int = 1
    task: str = "regression"
    device: str = "cpu"
    verbose: bool = True
    orig_features: list | None = None
    compressed_features: list | None = None
    cat_impute_values: dict | None = None


def align_frame(spec: PyTorchFoldSpec, df):
    """Route a frame through the template's fitted preprocessing.

    The held-out validation rows are split off *before* the template fits its
    preprocessing, so they arrive raw; this applies the same fitted transforms —
    categorical mappings, compressed-feature decompression, categorical NaN
    imputation. Idempotent, so it is safe on a frame that already went through the
    template's own preprocessing.

    Args:
        spec: the :class:`PyTorchFoldSpec` carrying the fitted alignment state.
        df: the frame to align (not mutated).

    Returns:
        The aligned copy.
    """
    from workbench.endpoints.inference import convert_categorical_types, decompress_features

    df = df.copy()
    # An empty frame (no validation_ids) has nothing to transform, and decompression
    # rejects a 0-row column.
    if df.empty:
        return df
    if spec.category_mappings:
        df, _ = convert_categorical_types(df, list(spec.category_mappings), spec.category_mappings)
    if spec.compressed_features and any(f in df.columns for f in spec.compressed_features):
        df, _ = decompress_features(df, spec.orig_features, spec.compressed_features)
    for col, value in (spec.cat_impute_values or {}).items():
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(value)
    return df


def train_pytorch_fold(
    spec: PyTorchFoldSpec, train_tensors, val_tensors, *, fold_idx: int = 0, train_sample_weight=None
):
    """Train one ensemble member on prepared tensors and return it.

    The single definition of "train one PyTorch tabular model" — used both to publish
    an ensemble and to score an HPO trial, so a searched config is evaluated exactly
    as it will be deployed: same per-fold seed offset, same architecture construction,
    same optimizer and early-stopping schedule.

    Args:
        spec: the fold-invariant :class:`PyTorchFoldSpec`.
        train_tensors: ``(x_cont, x_cat, y)`` from ``pytorch_utils.prepare_data``.
        val_tensors: ``(x_cont, x_cat, y)`` — drives early stopping.
        fold_idx: ensemble member index — offsets the seed so members differ.
        train_sample_weight: per-row loss weights for this fold's training rows. Validation is
            left unweighted, matching the XGBoost path, which weights ``train_idx`` only.

    Returns:
        tuple: ``(model, history)`` — the trained model and its training history.
    """
    import torch

    from workbench.endpoints.pytorch_utils import create_model, train_model

    hp = spec.hyperparameters
    hidden_layers = [int(x) for x in str(hp["layers"]).split("-")]

    torch.manual_seed(hp.get("seed", 42) + fold_idx)
    model = create_model(
        n_continuous=len(spec.continuous_cols),
        categorical_cardinalities=spec.categorical_cardinalities,
        hidden_layers=hidden_layers,
        n_outputs=spec.n_outputs,
        task=spec.task,
        dropout=hp["dropout"],
    )
    return train_model(
        model,
        *train_tensors,
        *val_tensors,
        task=spec.task,
        max_epochs=hp["max_epochs"],
        patience=hp["early_stopping_patience"],
        batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        loss=hp["loss"],
        device=spec.device,
        restore_best_weights=hp["restore_best_weights"],
        verbose=spec.verbose,
        train_sample_weight=train_sample_weight,
    )
