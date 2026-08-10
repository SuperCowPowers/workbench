"""TargetLandscape: target-vs-structure landscape analysis on top of a Proximity backend.

Owns analysis that depends on the "nearest-neighbor topology" of the reference set:
    - Duplicates (identical structure/features, possibly conflicting targets)
    - Activity cliffs (steep target gradients between distinct neighbors)
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

# Distance below which two rows are treated as the same point in the space.
COINCIDENT_EPS = 1e-6


class TargetLandscape:
    """Landscape analysis (duplicates, activity cliffs, isolated compounds,
    distribution stats) built on top of a Proximity backend.

    Per-row nearest-neighbor columns (`nn_distance`, `nn_id`, `nn_target`,
    `nn_target_diff`) are computed lazily on first method call and cached on the
    proximity model's reference DataFrame.

    Two task-shaped entry points sit above the general `target_gradients()` engine:
        - `duplicates()` — coincident rows, grouped by structure
        - `cliffs()` — steep gradients between *distinct* neighbors

    Run `duplicates()` first: coincident rows carry an unbounded gradient, so on a
    set that has duplicates they crowd out every genuine cliff.
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

        # n=2 so that self plus one true neighbor are always in range. A coincident
        # duplicate ties with the row at distance 0 and can sort ahead of it, so the
        # neighbor is "the first index that isn't this row" rather than index 1.
        X = self.prox._transform_features(df)
        distances, indices = self.prox.nn.kneighbors(X, n_neighbors=2)

        rows = np.arange(len(df))
        neighbor_slot = np.argmax(indices != rows[:, None], axis=1)
        nn_rows = indices[rows, neighbor_slot]

        df["nn_distance"] = distances[rows, neighbor_slot]
        df["nn_id"] = df.iloc[nn_rows][self.prox.id_column].values

        if self.prox.target and self.prox.target in df.columns:
            nn_target_values = df.iloc[nn_rows][self.prox.target].values
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

    def duplicates(self, min_spread: Optional[float] = None) -> pd.DataFrame:
        """Find groups of rows that occupy the same point in the space.

        Identical structures (or identical feature vectors) carrying different target
        values are a data-integrity problem rather than chemistry — the same input
        can't map to two outputs, so a model trained on them fits noise. Groups with
        zero spread are redundant rather than contradictory.

        A fingerprint backend treats stereoisomers as coincident: Morgan fingerprints
        don't encode chirality, so enantiomers with genuinely different activity land
        in the same group.

        Args:
            min_spread: Only return groups whose target spread (max - min) is at least
                this. If None, all coincident groups are returned.

        Returns:
            DataFrame with one row per group member, sorted by `group_spread`
            descending: group_id, <id_column>, <target>, group_size, group_spread,
            group_median. Empty DataFrame if the set has no coincident rows.
        """
        self._ensure_metrics()
        df = self.prox.df
        target = self.prox.target if self.prox.target in df.columns else None

        columns = [
            "group_id",
            self.prox.id_column,
            *([target, "group_spread", "group_median"] if target else []),
            "group_size",
        ]

        # Only rows whose nearest neighbor is coincident can belong to a group
        candidate_rows = np.where(df["nn_distance"].values < COINCIDENT_EPS)[0]
        if len(candidate_rows) == 0:
            return pd.DataFrame(columns=columns)

        # Coincidence is transitive, so each row's neighbor set is already the full
        # group — dedupe identical sets rather than merging connected components.
        X = self.prox._transform_features(df)
        _, neighbor_sets = self.prox.nn.radius_neighbors(X[candidate_rows], radius=COINCIDENT_EPS)

        seen = set()
        records = []
        for members in (frozenset(s.tolist()) for s in neighbor_sets):
            if members in seen:
                continue
            seen.add(members)

            member_rows = df.iloc[sorted(members)]
            group = {"group_id": len(seen) - 1, "group_size": len(members)}
            if target:
                values = member_rows[target]
                group["group_spread"] = values.max() - values.min()
                group["group_median"] = values.median()

            for _, row in member_rows.iterrows():
                member = {**group, self.prox.id_column: row[self.prox.id_column]}
                if target:
                    member[target] = row[target]
                records.append(member)

        result = pd.DataFrame(records, columns=columns)
        if target:
            if min_spread is not None:
                result = result[result["group_spread"] >= min_spread]
            result = result.sort_values(["group_spread", "group_id"], ascending=[False, True])
        return result.reset_index(drop=True)

    def cliffs(
        self,
        top_percent: float = 1.0,
        min_delta: Optional[float] = None,
        k_neighbors: int = 4,
    ) -> pd.DataFrame:
        """Find activity cliffs — steep target gradients between *distinct* neighbors.

        Coincident rows are excluded; they're duplicates rather than cliffs, and their
        unbounded gradient would otherwise crowd out every genuine cliff. Use
        `duplicates()` for those.

        Args:
            top_percent: Percentage of compounds with steepest gradients to return.
            min_delta: Minimum absolute target difference to consider. If None, defaults
                to target_range/100.
            k_neighbors: Number of neighbors used for median verification (default: 4).

        Returns:
            DataFrame of compounds with the steepest gradients, sorted descending.
        """
        return self.target_gradients(
            top_percent=top_percent,
            min_delta=min_delta,
            k_neighbors=k_neighbors,
            min_distance=COINCIDENT_EPS,
        )

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
        min_distance: Optional[float] = None,
    ) -> pd.DataFrame:
        """Find compounds with steep target gradients.

        The general engine — `duplicates()` and `cliffs()` are the task-shaped entry
        points and are usually what you want.

        Two-phase approach:
            1. Quick filter on cliff_score = (nn_target_diff / target_range) / nn_distance
            2. Verify with k-neighbor median to filter out cases where the nearest neighbor
               is itself the outlier.

        `cliff_score` is normalized by the target range, so thresholds carry across
        endpoints with different units.

        Args:
            top_percent: Percentage of compounds with steepest gradients to return.
            min_delta: Minimum absolute target difference to consider. If None, defaults
                to target_range/100.
            k_neighbors: Number of neighbors used for median verification (default: 4).
            only_coincident: If True, only return compounds whose nearest neighbor is
                effectively coincident (distance ~0).
            min_distance: Only consider compounds whose nearest neighbor is at least this
                far away. Pass `COINCIDENT_EPS` to exclude coincident rows.

        Returns:
            DataFrame of compounds with steepest gradients, sorted descending.
        """
        if self.prox.target is None:
            raise ValueError("target_gradients requires a Proximity backend with `target` set")
        if only_coincident and min_distance is not None:
            raise ValueError("only_coincident and min_distance are mutually exclusive")

        self._ensure_metrics()
        df = self.prox.df

        # Normalize by target range so cliff_score is comparable across endpoints
        target_scale = self.target_range if self.target_range and self.target_range > 0 else 1.0

        # Phase 1: quick filter on precomputed nearest neighbor
        candidates = df.copy()
        candidates["cliff_score"] = (candidates["nn_target_diff"] / target_scale) / (
            candidates["nn_distance"] + COINCIDENT_EPS
        )

        if min_delta is None:
            min_delta = target_scale / 100.0
        candidates = candidates[candidates["nn_target_diff"] >= min_delta]

        if only_coincident:
            candidates = candidates[candidates["nn_distance"] < COINCIDENT_EPS].copy()
        else:
            if min_distance is not None:
                candidates = candidates[candidates["nn_distance"] >= min_distance]
            if candidates.empty:
                return self._empty_gradients()
            percentile = 100 - top_percent
            threshold = np.percentile(candidates["cliff_score"], percentile)
            candidates = candidates[candidates["cliff_score"] >= threshold].copy()

        # Phase 2: verify with k-neighbor median (handles nearest-neighbor-is-outlier case)
        results = []
        for _, row in candidates.iterrows():
            cmpd_id = row[self.prox.id_column]
            cmpd_target = row[self.prox.target]

            nbrs = self.prox.neighbors(cmpd_id, n_neighbors=k_neighbors, include_self=False)

            # Skip the nearest neighbor: it's what scored the cliff in Phase 1, so the
            # median over the rest is an independent check
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
                        "cliff_score": row["cliff_score"],
                        "neighbor_median": neighbor_median,
                        "neighbor_median_diff": median_diff,
                    }
                )

        if not results:
            return self._empty_gradients()

        results_df = pd.DataFrame(results)
        return results_df.sort_values("cliff_score", ascending=False).reset_index(drop=True)

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

    def _empty_gradients(self) -> pd.DataFrame:
        """Empty result with the target_gradients column schema."""
        return pd.DataFrame(
            columns=[
                self.prox.id_column,
                self.prox.target,
                "nn_target",
                "nn_target_diff",
                "nn_distance",
                "cliff_score",
                "neighbor_median",
                "neighbor_median_diff",
            ]
        )
