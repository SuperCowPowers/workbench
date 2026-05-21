"""ActivityLandscape: chemical/feature-space landscape analysis on top of a Proximity backend.

Owns analysis that depends on the "nearest-neighbor topology" of the reference set:
    - Activity cliffs (steep target gradients between neighbors)
    - Isolated compounds (low similarity to anything)
    - Distribution stats over nearest-neighbor distance/similarity
    - 2D projection for visualization (delegates to the proximity backend)

Composition over inheritance: takes any Proximity backend (FingerprintProximity,
FeatureSpaceProximity) and lazily computes per-row nn_* columns on first access.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from workbench.algorithms.dataframe.proximity import Proximity

log = logging.getLogger("workbench")


class ActivityLandscape:
    """Landscape analysis (activity cliffs, isolated compounds, distribution stats)
    built on top of a Proximity backend.

    Per-row nearest-neighbor columns (`nn_distance`, `nn_id`, `nn_target`,
    `nn_target_diff`) are computed lazily on first method call and cached on the
    proximity model's reference DataFrame.
    """

    def __init__(self, prox: Proximity):
        """
        Args:
            prox: A Proximity backend (FingerprintProximity, FeatureSpaceProximity, ...).
                Must have `id_column` set; `target` is required for `target_gradients`.
        """
        self.prox = prox
        self._metrics_computed = False
        # Distance column for results — most analyses prefer "similarity" if the
        # backend produces it (FingerprintProximity); otherwise fall back to distance.
        self._has_similarity = hasattr(prox, "_add_similarity_column")  # FP-flavored
        self._proximity_col = "nn_similarity" if self._has_similarity else "nn_distance"

    # ------------------------------------------------------------------
    # Lazy precomputation of per-row nearest-neighbor columns
    # ------------------------------------------------------------------

    def _ensure_metrics(self) -> None:
        """Compute nn_distance / nn_id / nn_target / nn_target_diff on the proximity
        model's reference DataFrame. Idempotent — runs once.
        """
        if self._metrics_computed:
            return

        df = self.prox.df
        log.info("Precomputing landscape metrics...")

        # n=2 because index 0 is self
        X = self.prox._transform_features(df)
        distances, indices = self.prox.nn.kneighbors(X, n_neighbors=2)

        df["nn_distance"] = distances[:, 1]
        df["nn_id"] = df.iloc[indices[:, 1]][self.prox.id_column].values

        if self.prox.target and self.prox.target in df.columns:
            nn_target_values = df.iloc[indices[:, 1]][self.prox.target].values
            df["nn_target"] = nn_target_values
            df["nn_target_diff"] = np.abs(df[self.prox.target].values - nn_target_values)
            self.target_range = df[self.prox.target].max() - df[self.prox.target].min()
        else:
            self.target_range = None

        # FingerprintProximity-flavored: also expose similarity
        if self._has_similarity:
            df["nn_similarity"] = 1 - df["nn_distance"]

        self._metrics_computed = True
        log.info("Landscape metrics computed")

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def isolated(self, top_percent: float = 1.0) -> pd.DataFrame:
        """Find isolated compounds based on nearest-neighbor proximity.

        For similarity-based backends: low similarity to nearest neighbor.
        For distance-based backends: high distance to nearest neighbor.

        Args:
            top_percent: Percentage of most isolated compounds to return (e.g., 1.0 = top 1%)

        Returns:
            DataFrame of observations sorted by isolation (most isolated first).
        """
        self._ensure_metrics()
        df = self.prox.df

        if self._has_similarity:
            # Lower similarity = more isolated
            threshold = np.percentile(df["nn_similarity"], top_percent)
            isolated = df[df["nn_similarity"] <= threshold].copy()
            isolated = isolated.sort_values("nn_similarity", ascending=True)
        else:
            # Higher distance = more isolated
            threshold = np.percentile(df["nn_distance"], 100 - top_percent)
            isolated = df[df["nn_distance"] >= threshold].copy()
            isolated = isolated.sort_values("nn_distance", ascending=False)

        isolated = isolated.reset_index(drop=True)
        if self.prox.include_all_columns:
            return isolated
        return isolated[self._core_columns()]

    def target_gradients(
        self,
        top_percent: float = 1.0,
        min_delta: Optional[float] = None,
        k_neighbors: int = 4,
        only_coincident: bool = False,
    ) -> pd.DataFrame:
        """Find compounds with steep target gradients (activity cliffs / data quality issues).

        Two-phase approach:
            1. Quick filter: gradient = |target - nn_target| / nn_distance
            2. Verify with k-neighbor median to filter out cases where the nearest neighbor
               is itself the outlier.

        Args:
            top_percent: Percentage of compounds with steepest gradients to return.
            min_delta: Minimum absolute target difference to consider. If None, defaults
                to target_range/100.
            k_neighbors: Number of neighbors used for median verification (default: 4).
            only_coincident: If True, only return compounds whose nearest neighbor is
                effectively coincident (distance ~0).

        Returns:
            DataFrame of compounds with steepest gradients, sorted descending.
        """
        if self.prox.target is None:
            raise ValueError("target_gradients requires a Proximity backend with `target` set")

        self._ensure_metrics()
        df = self.prox.df
        epsilon = 1e-6

        # Phase 1: quick filter on precomputed nearest neighbor
        candidates = df.copy()
        candidates["gradient"] = candidates["nn_target_diff"] / (candidates["nn_distance"] + epsilon)

        if min_delta is None:
            min_delta = self.target_range / 100.0 if self.target_range and self.target_range > 0 else 0.0
        candidates = candidates[candidates["nn_target_diff"] >= min_delta]

        if only_coincident:
            candidates = candidates[candidates["nn_distance"] < epsilon].copy()
        else:
            percentile = 100 - top_percent
            threshold = np.percentile(candidates["gradient"], percentile)
            candidates = candidates[candidates["gradient"] >= threshold].copy()

        # Phase 2: verify with k-neighbor median (handles nearest-neighbor-is-outlier case)
        results = []
        for _, row in candidates.iterrows():
            cmpd_id = row[self.prox.id_column]
            cmpd_target = row[self.prox.target]

            nbrs = self.prox.neighbors(cmpd_id, n_neighbors=k_neighbors + 1, include_self=False)
            neighbor_median = (
                nbrs.iloc[1:][self.prox.target].median() if len(nbrs) > 1 else nbrs[self.prox.target].median()
            )
            median_diff = abs(cmpd_target - neighbor_median)

            if median_diff >= min_delta:
                results.append(
                    {
                        self.prox.id_column: cmpd_id,
                        self.prox.target: cmpd_target,
                        "nn_target": row["nn_target"],
                        "nn_target_diff": row["nn_target_diff"],
                        "nn_distance": row["nn_distance"],
                        "gradient": row["gradient"],
                        "neighbor_median": neighbor_median,
                        "neighbor_median_diff": median_diff,
                    }
                )

        if not results:
            return pd.DataFrame(
                columns=[
                    self.prox.id_column,
                    self.prox.target,
                    "nn_target",
                    "nn_target_diff",
                    "nn_distance",
                    "gradient",
                    "neighbor_median",
                    "neighbor_median_diff",
                ]
            )

        results_df = pd.DataFrame(results)
        return results_df.sort_values("gradient", ascending=False).reset_index(drop=True)

    def proximity_stats(self) -> pd.DataFrame:
        """Distribution stats for the nearest-neighbor proximity column.

        Returns:
            DataFrame with count, mean, std, and percentile statistics.
        """
        self._ensure_metrics()
        return (
            self.prox.df[self._proximity_col]
            .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .to_frame()
        )

    def project_2d(self) -> pd.DataFrame:
        """Compute a 2D UMAP projection of the reference set for visualization.

        Delegates to the proximity backend's `project_2d()` method. The reference
        DataFrame is updated in-place with 'x' / 'y' columns.

        Returns:
            The proximity model's reference DataFrame with 'x' / 'y' columns added.
        """
        if not hasattr(self.prox, "project_2d"):
            raise NotImplementedError(f"Proximity backend {type(self.prox).__name__} does not implement project_2d()")
        return self.prox.project_2d()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _core_columns(self) -> list:
        """Default columns to return when include_all_columns is False."""
        cols = [self.prox.id_column, self._proximity_col, "nn_id"]
        if self.prox.target:
            cols.extend([self.prox.target, "nn_target", "nn_target_diff"])
        return cols
