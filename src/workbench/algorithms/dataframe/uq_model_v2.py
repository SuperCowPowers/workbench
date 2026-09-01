"""UQModelV2: applicability-domain confidence from proximity neighbors.

V2 is a pure AD score — no model fitting, no ensemble std, no error model.
For each query, look at its k unique nearest neighbors and ask:

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

V2 reuses V1's proximity artifact (``uq_proximity.joblib``) when
both are present in a model bundle — no separate proximity file is written.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from workbench.algorithms.dataframe.proximity import Proximity

log = logging.getLogger("workbench")

__all__ = ["UQModelV2"]


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
    """Dedup raw neighbors to k unique per query (keeping the nearest)."""
    # neighbors() returns rows already sorted nearest-first within each query.
    # Drop duplicate (query, neighbor) pairs (caused by replicate measurements),
    # then take the top k per query.
    deduped = raw_nbrs.drop_duplicates(subset=[query_col, "neighbor_id"], keep="first")
    return deduped.groupby(query_col, group_keys=False).head(k)


def _with_distance(nbrs: pd.DataFrame) -> pd.DataFrame:
    """Normalize neighbor results to a 'distance' column across proximity backends.

    FingerprintProximity reports Tanimoto ``similarity``; feature-space backends
    report Euclidean ``distance`` directly.
    """
    if "similarity" in nbrs.columns:
        return nbrs.assign(distance=1.0 - nbrs["similarity"])
    return nbrs


class UQModelV2:
    """Pure applicability-domain UQ from proximity neighbors.

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
    _MAX_NEIGHBOR_REQUEST = 250

    @staticmethod
    def _label_coverage(prox: Proximity, target: str) -> float:
        """Fraction of the reference set carrying a label for this target."""
        return float(prox.df[target].notna().mean()) if target in prox.df.columns else 0.0

    @classmethod
    def _raw_request(cls, prox: Proximity, k: int, target: Optional[str] = None) -> int:
        """How many raw neighbors to request before dedup-to-k-unique.

        Scaled up by the target's label coverage so that k *labeled* neighbors are
        reachable in a sparse multi-target reference set. Capped at the reference
        set size minus one (to leave room for excluding self on training queries).
        """
        raw = k * cls._NEIGHBOR_OVERSHOOT
        if target is not None:
            coverage = cls._label_coverage(prox, target)
            if coverage > 0.0:
                raw = max(raw, int(np.ceil(k / coverage * cls._NEIGHBOR_OVERSHOOT)))
            raw = min(raw, cls._MAX_NEIGHBOR_REQUEST)
        return min(raw, max(1, len(prox.df) - 1))

    def _request_count(self, target: Optional[str] = None) -> int:
        """Instance-side wrapper over :meth:`_raw_request`."""
        return self._raw_request(self.prox, self.k, target)

    def __init__(
        self,
        prox: Proximity,
        k: int = 10,
        targets: Optional[Union[str, List[str]]] = None,
        distance_percentiles: Optional[List[float]] = None,
        variance_percentiles: Optional[Dict[str, List[float]]] = None,
        confidence_levels: Optional[List[float]] = None,
    ):
        """
        Args:
            prox: Proximity backend for neighborhood lookups, shared across targets.
            k: Number of unique nearest neighbors per query (default 10).
            targets: Target column(s) to model, defaulting to the backend's. The first
                is the primary — what predict() scores when no target is named.
            distance_percentiles: 0..100 percentiles of mean-neighbor-distance across
                the training set. Target-independent. Populated by fit() or load().
            variance_percentiles: Per target, the 0..100 percentiles of
                neighbor-target-std across the training set. Populated by fit() or load().
            confidence_levels: Coverage levels used for the neighbor-target quantile
                output (q_025..q_975). Default [0.50, 0.68, 0.80, 0.90, 0.95].
        """
        if prox is None:
            raise ValueError("UQModelV2 requires a non-None Proximity backend")

        self.prox = prox
        self.k = k
        if targets is None:
            self.targets = list(prox.targets)
        else:
            self.targets = [targets] if isinstance(targets, str) else list(targets)
        if not self.targets:
            raise ValueError("UQModelV2 requires at least one target column")
        self.distance_percentiles = list(distance_percentiles) if distance_percentiles is not None else None
        self.variance_percentiles = dict(variance_percentiles) if variance_percentiles is not None else None
        self.confidence_levels = confidence_levels or list(self.DEFAULT_CONFIDENCE_LEVELS)

    @property
    def primary_target(self) -> str:
        """The target predict() scores when the caller doesn't name one."""
        return self.targets[0]

    def _resolve_target(self, target: Optional[str]) -> str:
        """Name the target to act on, defaulting to the primary."""
        target = target or self.primary_target
        if self.variance_percentiles is not None and target not in self.variance_percentiles:
            raise RuntimeError(
                f"UQModelV2 has no calibration for '{target}' " f"(fitted: {sorted(self.variance_percentiles)})."
            )
        return target

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, prox: Proximity, k: int = 10, targets: Optional[Union[str, List[str]]] = None) -> "UQModelV2":
        """Compute reference percentile distributions of (mean_distance, target_std).

        For every training compound (rows in ``prox.df``), find its k unique nearest
        neighbors and record (mean_distance, neighbor_target_std). The 0..100
        percentiles of those two distributions become the reference for ranking
        query stats at predict time.

        Distance is target-independent, so its distribution is calibrated once over
        the whole reference set. Target spread is calibrated per target, over that
        target's labeled rows and their labeled neighbors.

        Args:
            prox: FingerprintProximity over the training set, with target column(s) set.
            k: Unique nearest-neighbor count for each query (default 10).
            targets: Target column(s) to calibrate, defaulting to the backend's.

        Returns:
            A fitted UQModelV2.
        """
        id_col = prox.id_column
        if targets is None:
            target_list = list(prox.targets)
        else:
            target_list = [targets] if isinstance(targets, str) else list(targets)
        if not target_list:
            raise ValueError("UQModelV2.fit requires at least one target column")
        train_ids = prox.df[id_col].unique().tolist()

        log.info(f"Fitting UQModelV2 on {len(train_ids)} training compounds " f"(k={k}, targets={target_list}) ...")

        # Distance calibration: one pass over the whole reference set. Over-request
        # to absorb replicate rows, capped at reference-set size minus one (proximity
        # has a broadcasting bug when n_neighbors > n_train, and self is excluded).
        raw_nbrs = prox.neighbors(train_ids, n_neighbors=cls._raw_request(prox, k), include_self=False)
        unique_nbrs = _with_distance(_unique_neighbors_per_query(raw_nbrs, query_col=id_col, k=k))
        mean_distances = unique_nbrs.groupby(id_col)["distance"].mean().dropna().to_numpy()
        if len(mean_distances) == 0:
            raise RuntimeError(
                "UQModelV2 fit produced no valid neighborhood stats. "
                "Check that the proximity contains at least k+1 training compounds."
            )
        distance_percentiles = [float(np.percentile(mean_distances, p)) for p in range(101)]
        log.info(
            f"  mean_distance:   min={mean_distances.min():.4f}, "
            f"median={np.median(mean_distances):.4f}, max={mean_distances.max():.4f}"
        )

        # Spread calibration, per target, over that target's labeled neighborhoods.
        variance_percentiles = {}
        for target in target_list:
            labeled_ids = prox.df.loc[prox.df[target].notna(), id_col].unique().tolist()
            if not labeled_ids:
                raise RuntimeError(f"UQModelV2 fit: no labeled rows for target '{target}'")
            raw = prox.neighbors(labeled_ids, n_neighbors=cls._raw_request(prox, k, target), include_self=False)
            labeled_nbrs = _unique_neighbors_per_query(raw[raw[target].notna()], query_col=id_col, k=k)
            target_stds = labeled_nbrs.groupby(id_col)[target].std().dropna().to_numpy()
            if len(target_stds) == 0:
                raise RuntimeError(
                    f"UQModelV2 fit: target '{target}' has no neighborhood with two labeled "
                    "neighbors; its labels are too sparse for applicability-domain UQ."
                )
            variance_percentiles[target] = [float(np.percentile(target_stds, p)) for p in range(101)]
            log.info(
                f"  target_std [{target}]: min={target_stds.min():.4f}, "
                f"median={np.median(target_stds):.4f}, max={target_stds.max():.4f}"
            )

        return cls(
            prox=prox,
            k=k,
            targets=target_list,
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
        target: Optional[str] = None,
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
            target: Which target to score, defaulting to the primary.

        Returns:
            DataFrame indexed by query id (or query_id for novel queries) with columns:
                confidence,
                neighbor_distance, neighbor_target_mean, neighbor_target_std,
                distance_percentile, variance_percentile,
                q_025, q_05, q_10, q_16, q_25, q_50, q_75, q_84, q_90, q_95, q_975
        """
        if self.distance_percentiles is None or not self.variance_percentiles:
            raise RuntimeError("UQModelV2 not fitted. Call .fit(...) first or .load(...).")

        target_col = self._resolve_target(target)

        # Auto-dispatch on query type (parallel to V1.predict). Cap n_neighbors
        # at the reference set size to avoid the proximity's broadcasting bug.
        # Capture the *expected* result identifiers so we can reindex the output
        # back to the input length even when the proximity silently drops rows
        # (e.g. unparseable SMILES). Callers do `df_val[col] = uq_out[col].values`
        # and expect len(uq_out) == len(query).
        n_request = self._request_count(target_col)
        if isinstance(query, pd.DataFrame):
            raw_nbrs = self.prox.neighbors_from_query_df(query, n_neighbors=n_request)
            query_col = "query_id"
            if "query_id" in query.columns:
                expected_index = list(query["query_id"].values)
            else:
                expected_index = list(range(len(query)))
        else:
            ids = list(query) if not isinstance(query, list) else query
            raw_nbrs = self.prox.neighbors(ids, n_neighbors=n_request, include_self=False)
            query_col = self.prox.id_column
            expected_index = ids

        if raw_nbrs.empty:
            # Nothing to score against — return an all-NaN frame matching the input
            return pd.DataFrame(
                index=pd.Index(expected_index, name=query_col),
                columns=self._result_columns(),
                dtype=float,
            )

        raw_nbrs = _with_distance(raw_nbrs)

        # Distance spans every neighbor — it describes the query's applicability
        # domain whether or not the neighbourhood happens to carry this target's label.
        unique_nbrs = _unique_neighbors_per_query(raw_nbrs, query_col=query_col, k=self.k)
        agg = unique_nbrs.groupby(query_col).agg(neighbor_distance=("distance", "mean"))

        # Target statistics come from the k nearest *labeled* neighbors.
        labeled_nbrs = _unique_neighbors_per_query(raw_nbrs[raw_nbrs[target_col].notna()], query_col, self.k)
        target_agg = labeled_nbrs.groupby(query_col).agg(
            neighbor_target_mean=(target_col, "mean"),
            neighbor_target_std=(target_col, "std"),
        )

        # Per-query neighbor-target quantiles (the V2 prediction intervals)
        # pandas groupby.quantile handles a single q at a time; build column-by-column
        for q_num, col_name in _NEIGHBOR_QUANTILES.items():
            target_agg[col_name] = labeled_nbrs.groupby(query_col)[target_col].quantile(q_num / 100.0)
        agg = agg.join(target_agg, how="outer")

        # Rank each query's mean_distance / target_std against stored distributions
        variance_percentiles = self.variance_percentiles[target_col]
        dist_pct = np.searchsorted(self.distance_percentiles, agg["neighbor_distance"].values, side="right") / len(
            self.distance_percentiles
        )
        # std is NaN for queries with under two labeled neighbors; treat as worst-case (pct=1)
        var_values = agg["neighbor_target_std"].fillna(np.inf).values
        var_pct = np.searchsorted(variance_percentiles, var_values, side="right") / len(variance_percentiles)
        dist_pct = np.clip(dist_pct, 0.0, 1.0)
        var_pct = np.clip(var_pct, 0.0, 1.0)

        agg["distance_percentile"] = dist_pct
        agg["variance_percentile"] = var_pct
        agg["confidence"] = np.clip((1.0 - dist_pct) * (1.0 - var_pct), 0.0, 1.0)

        # Reindex back to the input query's identifiers so that rows dropped by
        # the proximity (unparseable SMILES, missing ids) show up as NaN rather
        # than vanishing. Guarantees len(result) == len(query).
        return agg.reindex(expected_index)[self._result_columns()]

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
        if self.distance_percentiles is None or not self.variance_percentiles:
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
            "targets": self.targets,
            "confidence_levels": self.confidence_levels,
            "distance_percentiles": list(self.distance_percentiles),
            "variance_percentiles": {t: list(p) for t, p in self.variance_percentiles.items()},
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
            targets=metadata["targets"],
            distance_percentiles=metadata["distance_percentiles"],
            variance_percentiles=metadata["variance_percentiles"],
            confidence_levels=metadata.get("confidence_levels"),
        )
