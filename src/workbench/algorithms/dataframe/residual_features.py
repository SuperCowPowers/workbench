"""ResidualFeatures: neighborhood-derived features for UQ residual modeling.

Computes scalar features that describe each compound's local context in the
training set. These features feed into a residual estimator (RandomForestRegressor
inside UQModelV1) that predicts |residual| for an upstream ML model.

Validated by the 2025 J Chem Inf Model paper (PMC12848971): error models built
on [prediction, ensemble_variance, distance_to_training] outperform standard UQ
metrics across endpoints and under distribution shift.

Feature definitions per query compound:
    knn_distance     Mean distance to k nearest training neighbors.
                      Direct AD signal — large means novel chemistry.
    knn_target_mean  Mean target value of k nearest neighbors. Useful for
                      detecting attractor/cluster behavior.
    knn_target_std   Std of target values among k nearest neighbors. The key
                      signal for "dense neighborhood, heterogeneous labels"
                      failures (e.g. solubility censored-attractor case).
                      A query with knn_target_std >> 0 sits in a region where
                      the ensemble's tight agreement is misleadingly confident.
    local_pred_gap   prediction - knn_target_mean. Catches "model predicts the
                      cluster mean but neighbors are diverse" — only computed
                      when predictions are passed in.

Composition over inheritance: takes any Proximity backend.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
import logging

# Cross-module imports: workbench package path for library use; in-package
# sibling import when symlinked into a script bundle's model_script_utils/ package.
try:
    from workbench.algorithms.dataframe.proximity import Proximity
except ImportError:
    from .proximity import Proximity

log = logging.getLogger("workbench")


class ResidualFeatures:
    """Compute neighborhood-derived feature scalars for UQ residual modeling.

    Usage:
        prox = FingerprintProximity(train_df, id_column="id", target="logp")
        rf = ResidualFeatures(prox)

        # Training-time features for the validation set (ids already in df)
        val_feats = rf.compute(
            val_df["id"].tolist(),
            predictions=y_pred_val,
            k=10,
            training_only=True,
        )

        # Inference-time features for novel SMILES
        test_feats = rf.compute_from_query_df(
            pd.DataFrame({"smiles": test_smiles, "query_id": test_ids}),
            predictions=y_pred_test,
            k=10,
        )
    """

    def __init__(self, prox: Proximity):
        """
        Args:
            prox: Proximity backend with a target column set (required for
                knn_target_mean / knn_target_std).
        """
        if not prox.target:
            raise ValueError(
                "ResidualFeatures requires a Proximity backend with `target` set "
                "(knn_target_mean and knn_target_std need target values)"
            )
        self.prox = prox
        # Detect whether the backend returns 'similarity' (FingerprintProximity)
        # or 'distance' (FeatureSpaceProximity, etc.). We normalize to distance
        # internally so feature names are consistent across backends.
        self._distance_col = "similarity" if hasattr(prox, "_add_similarity_column") else "distance"

    # ------------------------------------------------------------------
    # Public feature-computation API
    # ------------------------------------------------------------------

    def compute(
        self,
        id_or_ids: Union[str, int, List],
        predictions: Optional[Union[np.ndarray, List[float]]] = None,
        k: int = 10,
        training_only: bool = False,
    ) -> pd.DataFrame:
        """Compute residual features for queries already in the reference set.

        Args:
            id_or_ids: Single ID or list of IDs.
            predictions: Optional model predictions for these queries (same order as ids).
                When provided, `local_pred_gap = prediction - knn_target_mean` is included.
            k: Number of nearest neighbors to consider (default: 10).
            training_only: If True, restrict neighbors to rows with `in_model=True`
                in the reference set (so cal/test rows are excluded as neighbors).
                Defaults to False.

        Returns:
            DataFrame indexed by query id with columns:
                knn_distance, knn_target_mean, knn_target_std, [local_pred_gap]
        """
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else id_or_ids

        # k+1 so self-exclusion leaves k true neighbors; include_self=False drops self-match
        nbrs = self.prox.neighbors(ids, n_neighbors=k + 1, include_self=False)

        return self._aggregate(
            nbrs,
            query_ids=ids,
            id_col=self.prox.id_column,
            predictions=predictions,
            training_only=training_only,
            k=k,
        )

    def compute_from_query_df(
        self,
        query_df: pd.DataFrame,
        predictions: Optional[Union[np.ndarray, List[float]]] = None,
        k: int = 10,
        training_only: bool = False,
    ) -> pd.DataFrame:
        """Compute residual features for novel queries (not in the reference set).

        Args:
            query_df: Novel-query DataFrame in the format the proximity backend expects
                (e.g. 'smiles' or 'fingerprint' column for FingerprintProximity).
                If 'query_id' is present it's used to label results; otherwise positional
                indices are used.
            predictions: Optional model predictions for these queries (same order as rows).
            k: Number of nearest neighbors to consider (default: 10).
            training_only: If True, restrict neighbors to rows with `in_model=True`.

        Returns:
            DataFrame indexed by query_id (or positional index) with columns:
                knn_distance, knn_target_mean, knn_target_std, [local_pred_gap]
        """
        # Novel queries are never self-matches; just ask for k neighbors directly
        nbrs = self.prox.neighbors_from_query_df(query_df, n_neighbors=k)

        if "query_id" in query_df.columns:
            query_ids = query_df["query_id"].tolist()
        else:
            query_ids = list(range(len(query_df)))

        return self._aggregate(
            nbrs,
            query_ids=query_ids,
            id_col="query_id",
            predictions=predictions,
            training_only=training_only,
            k=k,
        )

    # ------------------------------------------------------------------
    # Internal aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        nbrs: pd.DataFrame,
        query_ids: list,
        id_col: str,
        predictions,
        training_only: bool,
        k: int,
    ) -> pd.DataFrame:
        """Group neighbor rows by query and compute scalar features."""
        # Filter to training-only neighbors if requested
        if training_only:
            if "in_model" not in nbrs.columns:
                raise ValueError(
                    "training_only=True requires the proximity reference set to have "
                    "an `in_model` column (mark training rows with True)"
                )
            nbrs = nbrs[nbrs["in_model"]].copy()

        # Defensive: cap each query to its top-k neighbors. training_only filtering can
        # asymmetrically reduce per-query neighbor counts; this normalizes back to k.
        nbrs = nbrs.groupby(id_col, group_keys=False).head(k)

        # Normalize: produce a "distance"-style column regardless of backend
        if self._distance_col == "similarity":
            nbrs = nbrs.copy()
            nbrs["__distance__"] = 1 - nbrs["similarity"]
            dist_col = "__distance__"
        else:
            dist_col = "distance"

        # Aggregate per query
        target = self.prox.target
        agg = nbrs.groupby(id_col).agg(
            knn_distance=(dist_col, "mean"),
            knn_target_mean=(target, "mean"),
            knn_target_std=(target, "std"),
        )

        # Reindex to preserve caller's query order and surface missing queries as NaN
        agg = agg.reindex(query_ids)

        if predictions is not None:
            predictions = np.asarray(predictions, dtype=float)
            if len(predictions) != len(query_ids):
                raise ValueError(
                    f"predictions length ({len(predictions)}) does not match " f"number of queries ({len(query_ids)})"
                )
            agg["local_pred_gap"] = predictions - agg["knn_target_mean"].values

        agg.index.name = id_col
        return agg
