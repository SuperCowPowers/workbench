"""Aggregation nodes for ``MetaEndpointDAG``.

An aggregation node combines outputs from one or more upstream nodes
(``Endpoint`` or other ``AggregationNode`` instances) into a single DataFrame.
Two broad categories:

- **Column-union aggregators** (``Concat``): join feature outputs from
  parallel feature endpoints into a single wide row per ``id`` — used for
  feature-pipeline DAGs (e.g. ``[2D] + [3D] → Concat``).

- **Prediction aggregators** (``Mean``, ``WeightedMean``, ``Vote``, plus the
  ensemble-strategy ports ``ConfidenceWeighted``, ``InverseMaeWeighted``,
  ``ScaledConfidenceWeighted``, ``CalibratedConfidenceWeighted``): combine
  prediction columns from multiple predictor endpoints into a single
  ensemble prediction with confidence — used for ensemble combination.

Each node declares its input/output column contract so the DAG can be
validated statically before any inference runs.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from workbench.model_script_utils.meta_model_utils import (
    conf_weights_with_fallback,
    ensemble_confidence,
)


class AggregationNode:
    """Base class for DAG aggregation nodes.

    Subclasses implement ``apply()`` to combine upstream DataFrames and
    declare ``input_columns()`` / ``output_columns()`` for static DAG
    validation.

    All aggregation nodes carry a ``name`` (unique within a DAG) and an
    ``id_column`` used to join across upstream DataFrames.
    """

    def __init__(self, name: str, id_column: str = "id"):
        self.name = name
        self.id_column = id_column

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine upstream DataFrames into one. Subclasses must override."""
        raise NotImplementedError

    def input_columns(self, upstream_outputs: List[List[str]]) -> List[str]:
        """The columns this node expects across all upstream outputs.

        Default: union of all upstream output columns. Subclasses can
        narrow this if they only consume specific columns.
        """
        seen = set()
        cols: List[str] = []
        for upstream in upstream_outputs:
            for c in upstream:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        return cols

    def output_columns(self, upstream_outputs: List[List[str]]) -> List[str]:
        """The columns this node emits. Subclasses must override."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Column-union aggregator (feature pipelines)
# ---------------------------------------------------------------------------


class Concat(AggregationNode):
    """Column-union aggregator. Joins upstream DataFrames on ``id_column``.

    Use for feature-pipeline DAGs where parallel feature endpoints
    contribute disjoint feature column sets that need to be merged into a
    single wide row per ``id``.
    """

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        if not upstream:
            raise ValueError(f"Concat[{self.name}]: requires at least one upstream DataFrame")

        out = upstream[0]
        for i, df in enumerate(upstream[1:], start=1):
            new_cols = [c for c in df.columns if c == self.id_column or c not in out.columns]
            out = out.merge(df[new_cols], on=self.id_column, how="inner")
        return out

    def output_columns(self, upstream_outputs: List[List[str]]) -> List[str]:
        seen = set()
        cols: List[str] = []
        for upstream in upstream_outputs:
            for c in upstream:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        return cols


# ---------------------------------------------------------------------------
# Prediction aggregators (ensemble combination)
# ---------------------------------------------------------------------------


class _PredictionAggregator(AggregationNode):
    """Base for nodes that combine ``prediction``/``confidence`` columns
    from multiple predictor endpoints.

    Each upstream is expected to carry at minimum:
      - ``id_column``
      - ``prediction``
      - ``confidence`` (optional, depending on strategy)

    The output is a single DataFrame with ``id_column``, ``prediction``,
    ``prediction_std`` (ensemble disagreement), and ``confidence``.
    """

    OUTPUT_COLS = ["prediction", "prediction_std", "confidence"]

    def output_columns(self, upstream_outputs: List[List[str]]) -> List[str]:
        return [self.id_column] + self.OUTPUT_COLS

    def _stack(self, upstream: List[pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Align upstream frames on ``id_column`` and return (aligned ids,
        prediction matrix N×M, confidence matrix N×M)."""
        if not upstream:
            raise ValueError(f"{type(self).__name__}[{self.name}]: requires at least one upstream DataFrame")

        ids = upstream[0][[self.id_column]].copy()
        for df in upstream[1:]:
            ids = ids.merge(df[[self.id_column]], on=self.id_column, how="inner")

        preds = np.column_stack(
            [
                ids.merge(df[[self.id_column, "prediction"]], on=self.id_column)["prediction"].to_numpy()
                for df in upstream
            ]
        )
        confs = np.column_stack(
            [
                (
                    ids.merge(df[[self.id_column, "confidence"]], on=self.id_column)["confidence"].to_numpy()
                    if "confidence" in df.columns
                    else np.ones(len(ids))
                )
                for df in upstream
            ]
        )
        return ids, preds.astype(np.float64), confs.astype(np.float64)


class Mean(_PredictionAggregator):
    """Simple equal-weight mean of predictions."""

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        ids, preds, confs = self._stack(upstream)
        out = ids.copy()
        out["prediction"] = preds.mean(axis=1)
        out["prediction_std"] = preds.std(axis=1)
        out["confidence"] = confs.mean(axis=1)
        return out


class WeightedMean(_PredictionAggregator):
    """Static-weight mean — caller supplies one weight per upstream."""

    def __init__(self, name: str, weights: List[float], id_column: str = "id"):
        super().__init__(name, id_column=id_column)
        if not weights:
            raise ValueError("WeightedMean: weights must be a non-empty list")
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any():
            raise ValueError("WeightedMean: weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("WeightedMean: at least one weight must be positive")
        self.weights = w / w.sum()

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        if len(upstream) != len(self.weights):
            raise ValueError(
                f"WeightedMean[{self.name}]: got {len(upstream)} upstream frames " f"but {len(self.weights)} weights"
            )
        ids, preds, confs = self._stack(upstream)
        out = ids.copy()
        out["prediction"] = (preds * self.weights).sum(axis=1)
        out["prediction_std"] = preds.std(axis=1)
        out["confidence"] = (confs * self.weights).sum(axis=1)
        return out


class Vote(_PredictionAggregator):
    """Majority-vote aggregator for classifier predictions.

    Expects each upstream's ``prediction`` column to hold class labels
    (string or int). Output ``prediction`` is the most common label per
    row; ``prediction_std`` is 0 (placeholder for contract symmetry);
    ``confidence`` is the fraction of upstream models that voted for the
    winning label.
    """

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        if not upstream:
            raise ValueError(f"Vote[{self.name}]: requires at least one upstream DataFrame")

        ids = upstream[0][[self.id_column]].copy()
        for df in upstream[1:]:
            ids = ids.merge(df[[self.id_column]], on=self.id_column, how="inner")

        labels = pd.concat(
            [
                ids.merge(df[[self.id_column, "prediction"]], on=self.id_column)["prediction"].rename(f"_p{i}")
                for i, df in enumerate(upstream)
            ],
            axis=1,
        )

        modes = labels.mode(axis=1)[0]
        winner_share = (labels.eq(modes, axis=0)).sum(axis=1) / labels.shape[1]

        out = ids.copy()
        out["prediction"] = modes
        out["prediction_std"] = 0.0
        out["confidence"] = winner_share.to_numpy()
        return out


# ---------------------------------------------------------------------------
# Ensemble strategy nodes (ports of MetaModel's 5 strategies)
# ---------------------------------------------------------------------------


class _StrategyAggregator(_PredictionAggregator):
    """Shared infrastructure for the calibrated ensemble strategies.

    Each carries the per-model arrays (model_weights, corr_scale) and the
    blend factor (optimal_alpha) needed to compute calibrated ensemble
    confidence via :func:`ensemble_confidence`.
    """

    def __init__(
        self,
        name: str,
        model_weights: List[float],
        corr_scale: Optional[List[float]] = None,
        optimal_alpha: float = 0.5,
        id_column: str = "id",
    ):
        super().__init__(name, id_column=id_column)
        w = np.asarray(model_weights, dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"{type(self).__name__}: model_weights must be non-negative and sum to > 0")
        self.model_weights = w / w.sum()
        if corr_scale is None:
            self.corr_scale = np.ones(len(w))
        else:
            cs = np.asarray(corr_scale, dtype=np.float64)
            if cs.shape != w.shape:
                raise ValueError(f"{type(self).__name__}: corr_scale shape must match model_weights shape")
            self.corr_scale = cs
        self.optimal_alpha = float(optimal_alpha)

    def _check_arity(self, upstream: List[pd.DataFrame]) -> None:
        if len(upstream) != len(self.model_weights):
            raise ValueError(
                f"{type(self).__name__}[{self.name}]: got {len(upstream)} upstream frames "
                f"but {len(self.model_weights)} weights"
            )


class ConfidenceWeighted(_StrategyAggregator):
    """Per-row weights = upstream confidences (normalized).

    Falls back to static ``model_weights`` when row confidences sum to ~0.
    """

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        self._check_arity(upstream)
        ids, preds, confs = self._stack(upstream)
        weights = conf_weights_with_fallback(confs, self.model_weights)
        out = ids.copy()
        out["prediction"] = (preds * weights).sum(axis=1)
        out["prediction_std"] = preds.std(axis=1)
        out["confidence"] = ensemble_confidence(preds, confs, self.corr_scale, self.model_weights, self.optimal_alpha)
        return out


class InverseMaeWeighted(_StrategyAggregator):
    """Static per-model weights from inverse-MAE.

    The caller passes the inverse-MAE-derived weights directly via
    ``model_weights``. Identical to ``WeightedMean`` for the prediction
    column, but additionally computes calibrated ensemble confidence.
    """

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        self._check_arity(upstream)
        ids, preds, confs = self._stack(upstream)
        out = ids.copy()
        out["prediction"] = (preds * self.model_weights).sum(axis=1)
        out["prediction_std"] = preds.std(axis=1)
        out["confidence"] = ensemble_confidence(preds, confs, self.corr_scale, self.model_weights, self.optimal_alpha)
        return out


class ScaledConfidenceWeighted(_StrategyAggregator):
    """Per-row weights = ``model_weights × confidence`` (normalized).

    Often the top performer in practice — combines static MAE-derived
    weighting with per-row confidence scaling.
    """

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        self._check_arity(upstream)
        ids, preds, confs = self._stack(upstream)
        scaled = confs * self.model_weights
        weights = conf_weights_with_fallback(scaled, self.model_weights)
        out = ids.copy()
        out["prediction"] = (preds * weights).sum(axis=1)
        out["prediction_std"] = preds.std(axis=1)
        out["confidence"] = ensemble_confidence(preds, confs, self.corr_scale, self.model_weights, self.optimal_alpha)
        return out


class CalibratedConfidenceWeighted(_StrategyAggregator):
    """Per-row weights = ``confidence × |conf-error correlation|`` (normalized).

    Rewards models whose confidence actually predicts accuracy.
    """

    def apply(self, upstream: List[pd.DataFrame]) -> pd.DataFrame:
        self._check_arity(upstream)
        ids, preds, confs = self._stack(upstream)
        calibrated = confs * self.corr_scale
        weights = conf_weights_with_fallback(calibrated, self.model_weights)
        out = ids.copy()
        out["prediction"] = (preds * weights).sum(axis=1)
        out["prediction_std"] = preds.std(axis=1)
        out["confidence"] = ensemble_confidence(preds, confs, self.corr_scale, self.model_weights, self.optimal_alpha)
        return out
