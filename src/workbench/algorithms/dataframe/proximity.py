"""Proximity ABC: a swappable contract for neighbor-lookup backends.

Concrete subclasses (FingerprintProximity, FeatureSpaceProximity) provide different
similarity definitions but share this query contract so downstream analysis classes
(ActivityLandscape, ApplicabilityDomain) can be polymorphic over the backend.

The ABC enforces:
    - Both id-based and novel-query lookups
    - A canonical neighbor-result DataFrame shape: id, neighbor_id, distance,
      [target], [in_model], plus any backend-specific extras (e.g. similarity)
    - Shared reference attributes (id_column, target, df) for downstream consumption

What the ABC deliberately does NOT enforce:
    - The distance metric (Jaccard / Ruzicka / Euclidean — subclass detail)
    - The index data structure (ball_tree / sparse on-the-fly / KDTree — subclass detail)
    - The "novel query" input representation — query_df is structural; each subclass
      declares its own column requirements in the docstring
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import logging

# Set up logging
log = logging.getLogger("workbench")


class Proximity(ABC):
    """Abstract base for compound proximity backends."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        features: List[str],
        target: Optional[str] = None,
        include_all_columns: bool = False,
    ):
        """
        Initialize the Proximity class.

        Args:
            df: DataFrame containing the reference set for neighbor computations.
            id_column: Name of the column used as the identifier.
            features: List of feature column names used for neighbor computations.
            target: Name of the target column. Defaults to None.
            include_all_columns: Include all DataFrame columns in neighbor results.
                Defaults to False.
        """
        self.id_column = id_column
        self.features = features
        self.target = target
        self.include_all_columns = include_all_columns

        # Store the DataFrame (subclasses may filter/modify in _prepare_data)
        self.df = df.copy()

        # Subclass hooks
        self._prepare_data()
        self._build_model()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Prepare the reference DataFrame before building the model.

        Default: no-op. Subclasses can override to compute / validate / filter columns.
        """
        pass

    @abstractmethod
    def _build_model(self) -> None:
        """Build the underlying NN index.

        Must set `self.nn` to an object with sklearn-compatible
        kneighbors(X, n_neighbors) and radius_neighbors(X, radius) methods.
        """

    @abstractmethod
    def _transform_features(self, df: pd.DataFrame) -> Union[np.ndarray, "object"]:
        """Transform a DataFrame into the feature representation expected by self.nn.

        For id-based queries this is called with a slice of self.df. For novel
        queries this is called with the caller-supplied query_df.
        """

    # ------------------------------------------------------------------
    # Concrete neighbor query API (the ABC contract)
    # ------------------------------------------------------------------

    def neighbors(
        self,
        id_or_ids: Union[str, int, List[Union[str, int]]],
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """Look up neighbors for IDs already in the reference set.

        Args:
            id_or_ids: Single ID or list of IDs to look up.
            n_neighbors: Number of neighbors to return (ignored if radius is set).
            radius: If provided, find all neighbors within this distance.
            include_self: Whether to include self in results.

        Returns:
            DataFrame with columns: id_column, neighbor_id, distance, [target],
            [in_model], plus any backend-specific extras.
        """
        # Normalize to list
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else id_or_ids

        # Be tolerant of unknown IDs: backends can silently drop rows during
        # indexing (e.g. FingerprintProximity drops compounds whose SMILES
        # RDKit can't convert), and we'd rather log + skip than blow up the
        # caller. Upstream consumers (residual_features._aggregate, UQModelV1
        # ._stack_features) already turn missing-id rows into NaN/neutral
        # values via reindex + nan_to_num, so dropped IDs flow through
        # harmlessly. Symmetric benefit on the query side too — typos or
        # stale caches no longer crash the call.
        known_ids = set(self.df[self.id_column])
        missing_ids = [i for i in ids if i not in known_ids]
        if missing_ids:
            sample = missing_ids[:5]
            log.warning(
                f"proximity.neighbors: skipping {len(missing_ids)} ID(s) not in dataset "
                f"(sample: {sample}{'...' if len(missing_ids) > 5 else ''}). "
                "Usually means the backend dropped rows during indexing — check earlier "
                "logs for warnings like 'Failed to convert SMILES' or similar."
            )
            ids = [i for i in ids if i in known_ids]
            if not ids:
                # All IDs missing — return an empty frame rather than crashing in
                # the backend. Callers using _aggregate() will get all-NaN rows.
                return pd.DataFrame()

        # Filter to requested IDs and preserve order
        query_df = self.df[self.df[self.id_column].isin(ids)]
        query_df = query_df.set_index(self.id_column).loc[ids].reset_index()

        return self._neighbors_impl(
            query_df=query_df,
            query_ids=query_df[self.id_column].values,
            n_neighbors=n_neighbors,
            radius=radius,
            include_self=include_self,
            id_col_name=self.id_column,
        )

    def neighbors_from_query_df(
        self,
        query_df: pd.DataFrame,
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = None,
    ) -> pd.DataFrame:
        """Look up neighbors for novel queries (not yet in the reference set).

        Each subclass documents the required columns of query_df:
            - FingerprintProximity: 'smiles' column (or precomputed 'fingerprint')
            - FeatureSpaceProximity: the feature columns the model was built with

        Args:
            query_df: Novel-query DataFrame. If a 'query_id' column is present it's
                used to label results; otherwise positional indices are used.
            n_neighbors: Number of neighbors to return (ignored if radius is set).
            radius: If provided, find all neighbors within this distance.

        Returns:
            DataFrame with columns: query_id, neighbor_id, distance, [target],
            [in_model], plus any backend-specific extras.
        """
        # Determine query labels
        if "query_id" in query_df.columns:
            query_ids = query_df["query_id"].values
        else:
            query_ids = np.arange(len(query_df))

        # include_self isn't meaningful for novel queries
        return self._neighbors_impl(
            query_df=query_df,
            query_ids=query_ids,
            n_neighbors=n_neighbors,
            radius=radius,
            include_self=True,
            id_col_name="query_id",
        )

    # ------------------------------------------------------------------
    # Internal: shared neighbor-result construction
    # ------------------------------------------------------------------

    def _neighbors_impl(
        self,
        query_df: pd.DataFrame,
        query_ids: np.ndarray,
        n_neighbors: Optional[int],
        radius: Optional[float],
        include_self: bool,
        id_col_name: str,
    ) -> pd.DataFrame:
        """Shared backend for neighbors() and neighbors_from_query_df()."""
        # Transform query features (subclass-specific)
        X_query = self._transform_features(query_df)

        # Get neighbors from the backend NN index
        if radius is not None:
            distances, indices = self.nn.radius_neighbors(X_query, radius=radius)
            # Ragged arrays — concatenate
            flat_distances = np.concatenate(distances) if len(distances) else np.array([])
            flat_indices = np.concatenate(indices) if len(indices) else np.array([], dtype=int)
            repeat_counts = [len(d) for d in distances]
            query_ids_repeated = np.repeat(query_ids, repeat_counts)
        else:
            distances, indices = self.nn.kneighbors(X_query, n_neighbors=n_neighbors)
            flat_distances = distances.ravel()
            flat_indices = indices.ravel()
            query_ids_repeated = np.repeat(query_ids, n_neighbors)

        # Vectorized neighbor lookup
        neighbor_ids = self.df[self.id_column].values[flat_indices]

        # Filter self-hits if requested (only meaningful for id-based lookups)
        if not include_self:
            mask = neighbor_ids != query_ids_repeated
            flat_distances = flat_distances[mask]
            flat_indices = flat_indices[mask]
            query_ids_repeated = query_ids_repeated[mask]
            neighbor_ids = neighbor_ids[mask]

        # Clean near-zero distances
        flat_distances = np.where(flat_distances < 1e-6, 0.0, flat_distances)

        # Build the canonical result dict
        result = {
            id_col_name: query_ids_repeated,
            "neighbor_id": neighbor_ids,
            "distance": flat_distances,
        }

        # Add target if present
        if self.target and self.target in self.df.columns:
            result[self.target] = self.df[self.target].values[flat_indices]

        # Pass through prediction-related and in_model columns
        for col in self.df.columns:
            if col == "prediction" or "_proba" in col or "residual" in col or col == "in_model":
                result[col] = self.df[col].values[flat_indices]

        # Include all columns if requested
        if self.include_all_columns:
            neighbor_rows = self.df.iloc[flat_indices]
            for col in neighbor_rows.columns:
                if col not in result:
                    result[col] = neighbor_rows[col].values
            # Restore query / neighbor id columns
            result[id_col_name] = query_ids_repeated
            result["neighbor_id"] = neighbor_ids

        df_results = pd.DataFrame(result)
        df_results = df_results.sort_values([id_col_name, "distance"], ascending=[True, True])
        return df_results.reset_index(drop=True)
