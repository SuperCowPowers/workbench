"""UQModelV2: applicability-domain confidence from fingerprint proximity.

V2 is a pure AD score — no model fitting, no ensemble std, no error model.
For each query, look at its k unique nearest fingerprint neighbors and ask:

    1. Are they close?         (low mean Tanimoto distance)
    2. Do they agree on the    (low std of neighbor targets)
       target?

Confidence is high only when both are true:

    confidence = (1 - distance_percentile) * (1 - variance_percentile)

where each percentile is the rank of the query's stat against the training
set's empirical distribution.

Prediction intervals are derived directly from the k neighbors' target values
(q_05 / q_95 are the 5th/95th percentiles of those target values), centered
on the neighbor median — NOT on the model's prediction. This is intentional:
when the model disagrees with its neighbors, the marker sits outside the
neighbor-derived interval and that gap is itself the cliff diagnostic.

Compared to V0/V1:
    * V0 uses (prediction, std); no neighborhood. Misses AD violations.
    * V1 uses (prediction, std, neighbors) + RandomForest residual estimator.
    * V2 uses neighbors only; no model fitting. Most interpretable.

V2 is best for: "given training-similar compounds, how well-supported is
this query?" V2 is NOT a residual estimator — its confidence is a relative
ranking, not a calibrated P(correct) or error magnitude.

V2 reuses V1's fingerprint proximity artifact (``uq_proximity.joblib``) when
both are present in a model bundle — no separate proximity file is written.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Sibling-import fallback for the in-bundle case (matches V1 pattern)
try:
    from workbench.algorithms.dataframe.proximity import Proximity
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
except ImportError:
    from .proximity import Proximity  # noqa: F401
    from .fingerprint_proximity import FingerprintProximity  # noqa: F401

log = logging.getLogger("workbench")


# Coverage levels → quantile column names (matched to V0/V1 output schema)
_QUANTILE_COLUMNS = {
    0.50: ("q_25", "q_75"),
    0.68: ("q_16", "q_84"),
    0.80: ("q_10", "q_90"),
    0.90: ("q_05", "q_95"),
    0.95: ("q_025", "q_975"),
}

# All neighbor-target quantiles V2 emits (numeric percentile → column name)
_NEIGHBOR_QUANTILES = {
    2.5: "q_025",
    5: "q_05",
    10: "q_10",
    16: "q_16",
    25: "q_25",
    50: "q_50",
    75: "q_75",
    84: "q_84",
    90: "q_90",
    95: "q_95",
    97.5: "q_975",
}


def _unique_neighbors_per_query(raw_nbrs: pd.DataFrame, query_col: str, k: int) -> pd.DataFrame:
    """Dedup raw neighbors to k unique per query (keeping highest similarity)."""
    # neighbors() returns rows already sorted by similarity desc within each query.
    # Drop duplicate (query, neighbor) pairs (caused by replicate measurements),
    # then take the top k per query.
    deduped = raw_nbrs.drop_duplicates(subset=[query_col, "neighbor_id"], keep="first")
    return deduped.groupby(query_col, group_keys=False).head(k)


class UQModelV2:
    """Pure applicability-domain UQ from fingerprint proximity.

    Companion to :class:`UQModelV0` (isotonic) and :class:`UQModelV1`
    (proximity + RandomForest). Shares V0's / V1's ``.predict(query, predictions,
    prediction_std)`` signature for swap-compatibility, but ignores the
    ``predictions`` and ``prediction_std`` arguments — V2 derives confidence
    purely from the query's k nearest neighbors.

    Usage:
        # fit
        prox = FingerprintProximity(train_df, id_column="id", target="logp")
        uq2 = UQModelV2.fit(prox, k=10)

        # save / load (shares uq_proximity.joblib with V1)
        uq2.save(model_dir)
        uq2 = UQModelV2.load(model_dir)

        # predict
        out = uq2.predict(test_df[["smiles"]])
        # → confidence, neighbor_distance, neighbor_target_mean,
        #   neighbor_target_std, distance_percentile, variance_percentile,
        #   q_025, q_05, ..., q_50, ..., q_975
    """

    METADATA_FILENAME = "uq_metadata_v2.json"
    UQ_VERSION = "v2"

    DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]

    # Over-request factor when calling neighbors() to absorb FeatureSet
    # replicate rows (same molecule appearing multiple times). Empirically
    # 4× is enough for current open_admet data; 10× is safe overshoot.
    # Capped at the reference set size to avoid the proximity's
    # broadcasting bug when n_neighbors > n_train.
    _NEIGHBOR_OVERSHOOT = 10

    def _request_count(self, default_factor: int = None) -> int:
        """How many raw neighbors to request before dedup-to-k-unique.

        Capped at the proximity reference set size minus one (to leave room
        for excluding self on training-set queries).
        """
        factor = default_factor or self._NEIGHBOR_OVERSHOOT
        raw = self.k * factor
        max_available = max(1, len(self.prox.df) - 1)
        return min(raw, max_available)

    def __init__(
        self,
        prox: Proximity,
        k: int = 10,
        distance_percentiles: Optional[List[float]] = None,
        variance_percentiles: Optional[List[float]] = None,
        confidence_levels: Optional[List[float]] = None,
    ):
        """
        Args:
            prox: FingerprintProximity backend (target required) for neighborhood lookups.
            k: Number of unique nearest neighbors per query (default 10).
            distance_percentiles: 0..100 percentiles of mean-neighbor-distance across
                the training set. Populated by fit() or load().
            variance_percentiles: 0..100 percentiles of neighbor-target-std across the
                training set. Populated by fit() or load().
            confidence_levels: Coverage levels used for the neighbor-target quantile
                output (q_025..q_975). Default [0.50, 0.68, 0.80, 0.90, 0.95].
        """
        if prox is None:
            raise ValueError("UQModelV2 requires a non-None Proximity backend")
        if not getattr(prox, "target", None):
            raise ValueError("UQModelV2 requires the Proximity to have a target column set")

        self.prox = prox
        self.k = k
        self.distance_percentiles = list(distance_percentiles) if distance_percentiles is not None else None
        self.variance_percentiles = list(variance_percentiles) if variance_percentiles is not None else None
        self.confidence_levels = confidence_levels or list(self.DEFAULT_CONFIDENCE_LEVELS)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, prox: Proximity, k: int = 10) -> "UQModelV2":
        """Compute reference percentile distributions of (mean_distance, target_std).

        For every training compound (rows in ``prox.df``), find its k unique nearest
        neighbors and record (mean_distance, neighbor_target_std). The 0..100
        percentiles of those two distributions become the reference for ranking
        query stats at predict time.

        Args:
            prox: FingerprintProximity over the training set, with target column set.
            k: Unique nearest-neighbor count for each query (default 10).

        Returns:
            A fitted UQModelV2.
        """
        id_col = prox.id_column
        target_col = prox.target
        train_ids = prox.df[id_col].unique().tolist()

        log.info(f"Fitting UQModelV2 on {len(train_ids)} training compounds (k={k}) ...")

        # Bulk neighbor lookup with over-request to absorb replicate rows.
        # Cap at reference-set size minus one (proximity has a broadcasting bug
        # when n_neighbors > n_train, and we always exclude self anyway).
        n_request = min(k * cls._NEIGHBOR_OVERSHOOT, max(1, len(prox.df) - 1))
        raw_nbrs = prox.neighbors(
            train_ids,
            n_neighbors=n_request,
            include_self=False,
        )
        unique_nbrs = _unique_neighbors_per_query(raw_nbrs, query_col=id_col, k=k)

        # Per-query stats: mean distance, std of neighbor targets
        unique_nbrs = unique_nbrs.assign(distance=1.0 - unique_nbrs["similarity"])
        stats = unique_nbrs.groupby(id_col).agg(
            mean_distance=("distance", "mean"),
            target_std=(target_col, "std"),
        )

        # Reference distributions — handle NaN (e.g. compounds with <2 neighbors)
        mean_distances = stats["mean_distance"].dropna().to_numpy()
        target_stds = stats["target_std"].dropna().to_numpy()
        if len(mean_distances) == 0 or len(target_stds) == 0:
            raise RuntimeError(
                "UQModelV2 fit produced no valid neighborhood stats. "
                "Check that the proximity contains at least k+1 training compounds."
            )

        distance_percentiles = [float(np.percentile(mean_distances, p)) for p in range(101)]
        variance_percentiles = [float(np.percentile(target_stds, p)) for p in range(101)]

        log.info(
            f"  mean_distance:   min={mean_distances.min():.4f}, "
            f"median={np.median(mean_distances):.4f}, max={mean_distances.max():.4f}"
        )
        log.info(
            f"  target_std:      min={target_stds.min():.4f}, "
            f"median={np.median(target_stds):.4f}, max={target_stds.max():.4f}"
        )

        return cls(
            prox=prox,
            k=k,
            distance_percentiles=distance_percentiles,
            variance_percentiles=variance_percentiles,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        query: Union[List, pd.Series, np.ndarray, pd.DataFrame],
        predictions: Optional[Union[np.ndarray, pd.Series]] = None,
        prediction_std: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Compute V2 UQ outputs (AD confidence + neighbor-derived intervals).

        The ``predictions`` and ``prediction_std`` arguments are accepted for
        signature compatibility with V0/V1 but **ignored** in V2's math. V2
        derives everything from the query's k nearest neighbors.

        Args:
            query: IDs already in the proximity reference set (list/Series/array),
                or a DataFrame of novel queries (must contain 'smiles' or
                'fingerprint' for FingerprintProximity).
            predictions: Ignored. Accepted for V0/V1 compatibility.
            prediction_std: Ignored. Accepted for V0/V1 compatibility.

        Returns:
            DataFrame indexed by query id (or query_id for novel queries) with columns:
                confidence,
                neighbor_distance, neighbor_target_mean, neighbor_target_std,
                distance_percentile, variance_percentile,
                q_025, q_05, q_10, q_16, q_25, q_50, q_75, q_84, q_90, q_95, q_975
        """
        if self.distance_percentiles is None or self.variance_percentiles is None:
            raise RuntimeError("UQModelV2 not fitted. Call .fit(...) first or .load(...).")

        target_col = self.prox.target

        # Auto-dispatch on query type (parallel to V1.predict). Cap n_neighbors
        # at the reference set size to avoid the proximity's broadcasting bug.
        n_request = self._request_count()
        if isinstance(query, pd.DataFrame):
            raw_nbrs = self.prox.neighbors_from_query_df(query, n_neighbors=n_request)
            query_col = "query_id"
        else:
            ids = list(query) if not isinstance(query, list) else query
            raw_nbrs = self.prox.neighbors(ids, n_neighbors=n_request, include_self=False)
            query_col = self.prox.id_column

        if raw_nbrs.empty:
            # Nothing to score against — return an empty result with the right schema
            return pd.DataFrame(columns=self._result_columns())

        unique_nbrs = _unique_neighbors_per_query(raw_nbrs, query_col=query_col, k=self.k)
        unique_nbrs = unique_nbrs.assign(distance=1.0 - unique_nbrs["similarity"])

        # Per-query aggregates
        agg = unique_nbrs.groupby(query_col).agg(
            neighbor_distance=("distance", "mean"),
            neighbor_target_mean=(target_col, "mean"),
            neighbor_target_std=(target_col, "std"),
        )

        # Per-query neighbor-target quantiles (the V2 prediction intervals)
        # pandas groupby.quantile handles a single q at a time; build column-by-column
        for q_num, col_name in _NEIGHBOR_QUANTILES.items():
            agg[col_name] = unique_nbrs.groupby(query_col)[target_col].quantile(q_num / 100.0)

        # Rank each query's mean_distance / target_std against stored distributions
        dist_pct = (
            np.searchsorted(self.distance_percentiles, agg["neighbor_distance"].values, side="right")
            / len(self.distance_percentiles)
        )
        # std can be NaN for queries with <2 neighbors; treat as worst-case (pct=1)
        var_values = agg["neighbor_target_std"].fillna(np.inf).values
        var_pct = (
            np.searchsorted(self.variance_percentiles, var_values, side="right")
            / len(self.variance_percentiles)
        )
        dist_pct = np.clip(dist_pct, 0.0, 1.0)
        var_pct = np.clip(var_pct, 0.0, 1.0)

        agg["distance_percentile"] = dist_pct
        agg["variance_percentile"] = var_pct
        agg["confidence"] = np.clip((1.0 - dist_pct) * (1.0 - var_pct), 0.0, 1.0)

        return agg[self._result_columns()]

    @staticmethod
    def _result_columns() -> List[str]:
        """Canonical column order for the predict() output."""
        return [
            "confidence",
            "neighbor_distance",
            "neighbor_target_mean",
            "neighbor_target_std",
            "distance_percentile",
            "variance_percentile",
        ] + list(_NEIGHBOR_QUANTILES.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir: str, save_proximity: bool = True) -> None:
        """Save fitted state to a model directory.

        Writes ``uq_metadata_v2.json`` with the calibration percentiles.
        Shares ``uq_proximity.joblib`` with UQModelV1 — only writes it if not
        already present in the directory (i.e. V1 hasn't already saved it).

        For workbench-internal use where the proximity is rebuilt on demand
        from the source FeatureSet, set ``save_proximity=False`` to skip the
        proximity file entirely.
        """
        if self.distance_percentiles is None or self.variance_percentiles is None:
            raise RuntimeError("UQModelV2 not fitted; nothing to save.")
        os.makedirs(model_dir, exist_ok=True)

        # Shared proximity artifact with V1 — only write if not already there
        prox_path = os.path.join(model_dir, "uq_proximity.joblib")
        if save_proximity and not os.path.exists(prox_path):
            # Use V1's slim helper if available (avoid bloat); fall back to dumping prox
            try:
                from workbench.algorithms.dataframe.uq_model_v1 import UQModelV1
                slim = UQModelV1._slim_proximity(self.prox)
            except Exception:  # noqa: BLE001 — slim is an optimization, not required
                slim = self.prox
            joblib.dump(slim, prox_path)

        metadata = {
            "k": self.k,
            "confidence_levels": self.confidence_levels,
            "distance_percentiles": list(self.distance_percentiles),
            "variance_percentiles": list(self.variance_percentiles),
        }
        with open(os.path.join(model_dir, self.METADATA_FILENAME), "w") as fp:
            json.dump(metadata, fp, indent=2)

        log.info(f"Saved UQModelV2 to {model_dir}")

    @classmethod
    def load(cls, model_dir: str, prox: Optional[Proximity] = None) -> "UQModelV2":
        """Load a fitted UQModelV2 from disk.

        Args:
            model_dir: Directory containing uq_metadata_v2.json (and uq_proximity.joblib).
            prox: Proximity backend to use. If None, loads the embedded
                ``uq_proximity.joblib`` (shared with V1).

        Returns:
            A UQModelV2 ready to .predict(...).
        """
        metadata_path = os.path.join(model_dir, cls.METADATA_FILENAME)
        with open(metadata_path) as fp:
            metadata = json.load(fp)

        if prox is None:
            prox_path = os.path.join(model_dir, "uq_proximity.joblib")
            if not os.path.exists(prox_path):
                raise FileNotFoundError(
                    f"No proximity backend provided and no {prox_path} found. "
                    "Pass `prox=...` explicitly, or ensure V1 (or V2) saved its proximity."
                )
            prox = joblib.load(prox_path)

        return cls(
            prox=prox,
            k=metadata["k"],
            distance_percentiles=metadata["distance_percentiles"],
            variance_percentiles=metadata["variance_percentiles"],
            confidence_levels=metadata.get("confidence_levels"),
        )
