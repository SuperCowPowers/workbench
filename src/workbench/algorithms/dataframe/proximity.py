import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Union
import logging

# Set up logging
log = logging.getLogger("workbench")


class Proximity:
    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        features: List[str],
        target: Optional[str] = None,
        track_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the Proximity class.

        Args:
            df: DataFrame containing data for neighbor computations.
            id_column: Name of the column used as the identifier.
            features: List of feature column names to be used for neighbor computations.
            target: Name of the target column. Defaults to None.
            track_columns: Additional columns to track in results. Defaults to None.
        """
        self.id_column = id_column
        self.target = target
        self.track_columns = track_columns or []

        # Filter out non-numeric features
        self.features = self._validate_features(df, features)

        # Drop NaN rows and set up DataFrame
        self.df = df.dropna(subset=self.features).copy()

        # Compute target range if target is provided
        self.target_range = None
        if self.target and self.target in self.df.columns:
            self.target_range = self.df[self.target].max() - self.df[self.target].min()

        # Build the proximity model
        self._build_model()

        # Precompute landscape metrics
        self._precompute_metrics()

    def isolated(self, top_percent: float = 1.0) -> pd.DataFrame:
        """
        Find isolated data points based on distance to nearest neighbor.

        Args:
            top_percent: Percentage of most isolated data points to return (e.g., 1.0 returns top 1%)

        Returns:
            DataFrame of observations above the percentile threshold, sorted by distance (descending)
        """
        percentile = 100 - top_percent
        threshold = np.percentile(self.df["nn_distance"], percentile)
        isolated = self.df[self.df["nn_distance"] >= threshold].copy()
        return isolated.sort_values("nn_distance", ascending=False).reset_index(drop=True)

    def target_gradients(
        self,
        top_percent: float = 1.0,
        min_delta: Optional[float] = None,
        k_neighbors: int = 4,
        only_coincident: bool = False,
    ) -> pd.DataFrame:
        """
        Find compounds with steep target gradients (data quality issues and activity cliffs).

        Uses a two-phase approach:
        1. Quick filter using nearest neighbor gradient
        2. Verify using k-neighbor median to handle cases where the nearest neighbor is the outlier

        Args:
            top_percent: Percentage of compounds with steepest gradients to return (e.g., 1.0 = top 1%)
            min_delta: Minimum absolute target difference to consider. If None, defaults to target_range/100
            k_neighbors: Number of neighbors to use for median calculation (default: 4)
            only_coincident: If True, only consider compounds that are coincident (default: False)

        Returns:
            DataFrame of compounds with steepest gradients, sorted by gradient (descending)
        """
        if self.target is None:
            raise ValueError("Target column must be specified")

        epsilon = 1e-5

        # Phase 1: Quick filter using precomputed nearest neighbor
        candidates = self.df.copy()
        candidates["gradient"] = candidates["nn_target_diff"] / (candidates["nn_distance"] + epsilon)

        # Apply min_delta
        if min_delta is None:
            min_delta = self.target_range / 100.0 if self.target_range > 0 else 0.0
        candidates = candidates[candidates["nn_target_diff"] >= min_delta]

        # Filter based on mode
        if only_coincident:
            # Only keep coincident points (nn_distance ~= 0)
            candidates = candidates[candidates["nn_distance"] < epsilon].copy()
        else:
            # Get top X% by initial gradient
            percentile = 100 - top_percent
            threshold = np.percentile(candidates["gradient"], percentile)
            candidates = candidates[candidates["gradient"] >= threshold].copy()

        # Phase 2: Verify with k-neighbor median to filter out cases where nearest neighbor is the outlier
        results = []
        for _, row in candidates.iterrows():
            cmpd_id = row[self.id_column]
            cmpd_target = row[self.target]

            # Get k nearest neighbors (excluding self)
            nbrs = self.neighbors(cmpd_id, n_neighbors=k_neighbors, include_self=False)

            # Calculate median target of k neighbors, excluding the nearest neighbor (index 0)
            neighbor_median = nbrs.iloc[1:k_neighbors][self.target].median()
            median_diff = abs(cmpd_target - neighbor_median)

            # Only keep if compound differs from neighborhood median
            # This filters out cases where the nearest neighbor is the outlier
            if median_diff >= min_delta:
                results.append(
                    {
                        self.id_column: cmpd_id,
                        self.target: cmpd_target,
                        "nn_target": row["nn_target"],
                        "nn_target_diff": row["nn_target_diff"],
                        "nn_distance": row["nn_distance"],
                        "gradient": row["gradient"],  # Keep Phase 1 gradient
                        "neighbor_median": neighbor_median,
                        "neighbor_median_diff": median_diff,
                    }
                )

        # Handle empty results
        if not results:
            return pd.DataFrame(
                columns=[
                    self.id_column,
                    self.target,
                    "neighbor_median",
                    "neighbor_median_diff",
                    "mean_distance",
                    "gradient",
                ]
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("gradient", ascending=False).reset_index(drop=True)
        return results_df

    def neighbors(
        self,
        id_or_ids: Union[str, int, List[Union[str, int]]],
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """
        Return neighbors for ID(s) from the existing dataset.

        Args:
            id_or_ids: Single ID or list of IDs to look up
            n_neighbors: Number of neighbors to return (default: 5, ignored if radius is set)
            radius: If provided, find all neighbors within this radius
            include_self: Whether to include self in results (default: True)

        Returns:
            DataFrame containing neighbors and distances
        """
        # Normalize to list
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else id_or_ids

        # Validate IDs exist
        missing_ids = set(ids) - set(self.df[self.id_column])
        if missing_ids:
            raise ValueError(f"IDs not found in dataset: {missing_ids}")

        # Filter to requested IDs and preserve order
        query_df = self.df[self.df[self.id_column].isin(ids)]
        query_df = query_df.set_index(self.id_column).loc[ids].reset_index()

        # Transform query features
        X_query = self.scaler.transform(query_df[self.features])

        # Get neighbors
        if radius is not None:
            distances, indices = self.nn.radius_neighbors(X_query, radius=radius)
        else:
            distances, indices = self.nn.kneighbors(X_query, n_neighbors=n_neighbors)

        # Build results
        results = []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            query_id = query_df.iloc[i][self.id_column]

            for neighbor_idx, dist in zip(nbrs, dists):
                neighbor_id = self.df.iloc[neighbor_idx][self.id_column]

                # Skip self if requested
                if not include_self and neighbor_id == query_id:
                    continue

                results.append(self._build_neighbor_result(query_id=query_id, neighbor_idx=neighbor_idx, distance=dist))

        df_results = pd.DataFrame(results)
        df_results["is_self"] = df_results["neighbor_id"] == df_results[self.id_column]
        df_results = df_results.sort_values([self.id_column, "is_self", "distance"], ascending=[True, False, True])
        return df_results.drop("is_self", axis=1).reset_index(drop=True)

    def _validate_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove non-numeric features and log warnings."""
        non_numeric = [f for f in features if f not in df.select_dtypes(include=["number"]).columns]
        if non_numeric:
            log.warning(f"Non-numeric features {non_numeric} aren't currently supported, excluding them")
        return [f for f in features if f not in non_numeric]

    def _build_model(self) -> None:
        """Standardize features and fit Nearest Neighbors model."""
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self.df[self.features])
        self.nn = NearestNeighbors().fit(X)

    def _precompute_metrics(self, n_neighbors: int = 10) -> None:
        """
        Precompute landscape metrics for all compounds.

        Adds columns to self.df:
        - nn_distance: Distance to nearest neighbor
        - nn_id: ID of nearest neighbor

        If target is specified, also adds:
        - nn_target: Target value of nearest neighbor
        - nn_target_diff: Absolute difference from nearest neighbor target
        """
        log.info("Precomputing proximity metrics...")

        # Make sure n_neighbors isn't greater than dataset size
        n_neighbors = min(n_neighbors, len(self.df) - 1)

        # Get nearest neighbors for all points (including self)
        X = self.scaler.transform(self.df[self.features])
        distances, indices = self.nn.kneighbors(X, n_neighbors=2)  # Just need nearest neighbor

        # Extract nearest neighbor (index 1, since index 0 is self)
        self.df["nn_distance"] = distances[:, 1]
        self.df["nn_id"] = self.df.iloc[indices[:, 1]][self.id_column].values

        # If target exists, compute target-based metrics
        if self.target and self.target in self.df.columns:
            # Get target values for nearest neighbor
            nn_target_values = self.df.iloc[indices[:, 1]][self.target].values
            self.df["nn_target"] = nn_target_values
            self.df["nn_target_diff"] = np.abs(self.df[self.target].values - nn_target_values)

            # Precompute target range for min_delta default
            self.target_range = self.df[self.target].max() - self.df[self.target].min()

        log.info("Proximity metrics precomputed successfully")

    def _build_neighbor_result(self, query_id, neighbor_idx: int, distance: float) -> Dict:
        """
        Build a result dictionary for a single neighbor.

        Args:
            query_id: ID of the query point
            neighbor_idx: Index of the neighbor in the original DataFrame
            distance: Distance between query and neighbor

        Returns:
            Dictionary containing neighbor information
        """
        neighbor_row = self.df.iloc[neighbor_idx]
        neighbor_id = neighbor_row[self.id_column]

        # Start with basic info
        result = {
            self.id_column: query_id,
            "neighbor_id": neighbor_id,
            "distance": 0.0 if distance < 1e-5 else distance,
        }

        # Add target if present
        if self.target and self.target in self.df.columns:
            result[self.target] = neighbor_row[self.target]

        # Add tracked columns
        for col in self.track_columns:
            if col in self.df.columns:
                result[col] = neighbor_row[col]

        # Add prediction/probability columns if they exist
        for col in self.df.columns:
            if col == "prediction" or "_proba" in col or "residual" in col or col == "in_model":
                result[col] = neighbor_row[col]

        return result


# Testing the Proximity class
if __name__ == "__main__":

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create a sample DataFrame
    data = {
        "ID": [1, 2, 3, 4, 5],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "Feature3": [2.5, 2.4, 2.3, 2.3, np.nan],
    }
    df = pd.DataFrame(data)

    # Test the Proximity class
    features = ["Feature1", "Feature2", "Feature3"]
    prox = Proximity(df, id_column="ID", features=features)
    print(prox.neighbors(1, n_neighbors=2))

    # Test the neighbors method with radius
    print(prox.neighbors(1, radius=2.0))

    # Test with Features list
    prox = Proximity(df, id_column="ID", features=["Feature1"])
    print(prox.neighbors(1))

    # Create a sample DataFrame
    data = {
        "foo_id": ["a", "b", "c", "d", "e"],  # Testing string IDs
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)

    # Test with String Ids
    prox = Proximity(
        df,
        id_column="foo_id",
        features=["Feature1", "Feature2"],
        target="target",
        track_columns=["Feature1", "Feature2"],
    )
    print(prox.neighbors(["a", "b"]))

    # Test duplicate IDs
    data = {
        "foo_id": ["a", "b", "c", "d", "d"],  # Duplicate ID (d)
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)
    prox = Proximity(df, id_column="foo_id", features=["Feature1", "Feature2"], target="target")
    print(df.equals(prox.df))

    # Test with a categorical feature
    from workbench.api import FeatureSet, Model

    fs = FeatureSet("aqsol_features")
    model = Model("aqsol-regression")
    features = model.features()
    df = fs.pull_dataframe()
    prox = Proximity(
        df, id_column=fs.id_column, features=model.features(), target=model.target(), track_columns=features
    )
    print(prox.neighbors(df[fs.id_column].tolist()[:3]))

    print("\n" + "=" * 80)
    print("Testing isolated_compounds...")
    print("=" * 80)

    # Test isolated data in the top 1%
    isolated_1pct = prox.isolated(top_percent=1.0)
    print(f"\nTop 1% most isolated compounds (n={len(isolated_1pct)}):")
    print(isolated_1pct[[fs.id_column, "nn_distance", "nn_id"]].head(10))

    # Test isolated data in the top 5%
    isolated_5pct = prox.isolated(top_percent=5.0)
    print(f"\nTop 5% most isolated compounds (n={len(isolated_5pct)}):")
    print(isolated_5pct[[fs.id_column, "nn_distance", "nn_id"]].head(10))

    print("\n" + "=" * 80)
    print("Testing target_gradients...")
    print("=" * 80)

    # Test with different parameters
    gradients_1pct = prox.target_gradients(top_percent=1.0, min_delta=1.0)
    print(f"\nTop 1% target gradients (min_delta=5.0) (n={len(gradients_1pct)}):")
    print(
        gradients_1pct[
            [fs.id_column, model.target(), "neighbor_median", "neighbor_median_diff", "mean_distance", "gradient"]
        ].head(10)
    )

    gradients_5pct = prox.target_gradients(top_percent=5.0, min_delta=5.0)
    print(f"\nTop 5% target gradients (min_delta=5.0) (n={len(gradients_5pct)}):")
    print(
        gradients_5pct[
            [fs.id_column, model.target(), "neighbor_median", "neighbor_median_diff", "mean_distance", "gradient"]
        ].head(10)
    )
