"""Per-target prediction columns for multi-target models.

A multi-target model predicts several targets at once and carries each in its own columns:
``{target}_pred``, ``{target}_pred_std``, and the UQ outputs ``{target}_confidence`` /
``{target}_expected_residual`` / ``{target}_q_*``. The first target is also aliased to the
unprefixed ``prediction`` / ``prediction_std`` and UQ names, which is what single-target
consumers read.

The per-target columns are the source of truth; the unprefixed aliases are a convenience.
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

# UQ output columns, as the UQ models name them before per-target prefixing
UQ_COLUMN_PREFIXES = ("q_",)
UQ_COLUMN_NAMES = ("confidence", "expected_residual")


def per_target_predictions(
    target_columns: Sequence[str],
    preds: np.ndarray,
    ensemble_stack: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-target prediction and spread columns, in the model's space.

    Pure: it reads arrays and returns columns, leaving each caller to assign them however
    its frame is shaped (masked rows at inference, whole columns in training).
    ``report_targets`` moves the predictions to the reported scale once UQ has been scored.

    Args:
        target_columns: Target columns, in the model's output order.
        preds: ``(rows, targets)`` point predictions.
        ensemble_stack: ``(members, rows, targets)`` member predictions for the spread.
            None (a single model) gives a zero spread.

    Returns:
        A frame of ``{target}_pred`` and ``{target}_pred_std`` columns.
    """
    columns = {}
    for target_index, target in enumerate(target_columns):
        columns[f"{target}_pred"] = preds[:, target_index]
        if ensemble_stack is None:
            columns[f"{target}_pred_std"] = np.zeros(len(preds))
        else:
            columns[f"{target}_pred_std"] = np.std(ensemble_stack[:, :, target_index], axis=0)
    return pd.DataFrame(columns)


def alias_primary_target(df: pd.DataFrame, primary: str, uq_names: Sequence[str] = ()) -> None:
    """Copy the primary target's columns onto the unprefixed names, in place.

    Args:
        df: Frame holding the per-target columns.
        primary: The target to alias.
        uq_names: UQ column names to alias alongside the prediction columns.
    """
    df["prediction"] = df[f"{primary}_pred"]
    df["prediction_std"] = df[f"{primary}_pred_std"]
    for name in uq_names:
        if f"{primary}_{name}" in df.columns:
            df[name] = df[f"{primary}_{name}"]


def apply_per_target_uq(
    df: pd.DataFrame,
    uq_outs: dict,
    primary: str,
    masks: Union[None, np.ndarray, pd.Series, dict] = None,
) -> None:
    """Write ``{target}_``-prefixed UQ columns, aliasing the primary target's, in place.

    Args:
        df: Frame to write into.
        uq_outs: ``{target: frame}`` of UQ columns, covering the rows ``masks`` selects.
        primary: The target whose columns are also written unprefixed.
        masks: Row selector — one boolean array for every target, ``{target: array}`` for a
            per-target one, or None to write whole columns.
    """
    for target, uq_out in uq_outs.items():
        mask = masks.get(target) if isinstance(masks, dict) else masks
        for col in uq_out.columns:
            prefixed = f"{target}_{col}"
            if mask is None:
                df[prefixed] = uq_out[col].values
            else:
                df[prefixed] = np.nan
                df.loc[mask, prefixed] = uq_out[col].values
            if target == primary:
                df[col] = df[prefixed]


def report_targets(df: pd.DataFrame, adapters: dict, primary: str, truth: bool = False) -> None:
    """Move every target's prediction and quantiles to the reported scale, in place.

    Each target's ``{target}_pred`` and ``{target}_q_*`` columns go through that target's
    adapter, and the primary's unprefixed aliases go through the primary's.

    Args:
        df: Frame of model-space columns.
        adapters: ``{target: TargetAdapter}``.
        primary: The target the unprefixed ``prediction`` and ``q_*`` columns belong to.
        truth: True when the frame's target columns are in the model's space too (training
            rows). Held-out and inference rows carry original-scale targets.
    """
    for target, adapter in adapters.items():
        adapter.report(df, prediction=f"{target}_pred", prefix=f"{target}_", truth=target if truth else None)
    adapters[primary].report(df)


def multi_target_output_columns(df: pd.DataFrame, id_column: str, target_columns: Sequence[str]) -> list:
    """The saved-prediction column order for a multi-target model.

    Ids, the truth columns, per-target predictions and UQ, and the primary target's
    unprefixed aliases. Columns the frame does not carry are dropped, so one order serves
    both the out-of-fold and held-out files.

    Args:
        df: The frame being saved.
        id_column: The model's id column.
        target_columns: Target columns, in the model's output order.

    Returns:
        The ordered column names present in ``df``.
    """
    uq_names = [c for c in df.columns if c.startswith(UQ_COLUMN_PREFIXES) or c in UQ_COLUMN_NAMES]
    columns = [id_column] + list(target_columns)
    columns += [f"{t}_pred" for t in target_columns] + [f"{t}_pred_std" for t in target_columns]
    columns += ["prediction", "prediction_std"]
    columns += uq_names
    columns += [f"{t}_{c}" for t in target_columns for c in uq_names]
    columns += [c for c in df.columns if c.endswith("_proba")]
    return [c for c in columns if c in df.columns]
