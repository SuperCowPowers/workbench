import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union, List, Optional
import logging

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity
from workbench.algorithms.dataframe.projection_2d import Projection2D
from workbench.utils.chem_utils.fingerprints import compute_morgan_fingerprints

# Set up logging
log = logging.getLogger("workbench")


class FingerprintProximity(Proximity):
    """Proximity computations for binary fingerprints using Tanimoto similarity.

    Note: Tanimoto similarity is equivalent to Jaccard similarity for binary vectors.
    Tanimoto(A, B) = |A ∩ B| / |A ∪ B|
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        fingerprint_column: Optional[str] = None,
        target: Optional[str] = None,
        include_all_columns: bool = False,
        radius: int = 2,
        n_bits: int = 1024,
        counts: bool = False,
    ) -> None:
        """
        Initialize the FingerprintProximity class for binary fingerprint similarity.

        Args:
            df: DataFrame containing fingerprints or SMILES.
            id_column: Name of the column used as an identifier.
            fingerprint_column: Name of the column containing fingerprints (bit strings).
                If None, looks for existing "fingerprint" column or computes from SMILES.
            target: Name of the target column. Defaults to None.
            include_all_columns: Include all DataFrame columns in neighbor results. Defaults to False.
            radius: Radius for Morgan fingerprint computation (default: 2).
            n_bits: Number of bits for fingerprint (default: 1024).
            counts: Whether to use count simulation (default: False).
        """
        # Store fingerprint computation parameters
        self._fp_radius = radius
        self._fp_n_bits = n_bits
        self._fp_counts = counts

        # Store the requested fingerprint column (may be None)
        self._fingerprint_column_arg = fingerprint_column

        # Determine fingerprint column name (but don't compute yet - that happens in _prepare_data)
        self.fingerprint_column = self._resolve_fingerprint_column_name(df, fingerprint_column)

        # Call parent constructor with fingerprint_column as the only "feature"
        super().__init__(
            df,
            id_column=id_column,
            features=[self.fingerprint_column],
            target=target,
            include_all_columns=include_all_columns,
        )

    @staticmethod
    def _resolve_fingerprint_column_name(df: pd.DataFrame, fingerprint_column: Optional[str]) -> str:
        """
        Determine the fingerprint column name, validating it exists or can be computed.

        Args:
            df: Input DataFrame.
            fingerprint_column: Explicitly specified fingerprint column, or None.

        Returns:
            Name of the fingerprint column to use.

        Raises:
            ValueError: If no fingerprint column exists and no SMILES column found.
        """
        # If explicitly provided, validate it exists
        if fingerprint_column is not None:
            if fingerprint_column not in df.columns:
                raise ValueError(f"Fingerprint column '{fingerprint_column}' not found in DataFrame")
            return fingerprint_column

        # Check for existing "fingerprint" column
        if "fingerprint" in df.columns:
            log.info("Using existing 'fingerprint' column")
            return "fingerprint"

        # Will need to compute from SMILES - validate SMILES column exists
        smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
        if smiles_column is None:
            raise ValueError(
                "No fingerprint column provided and no SMILES column found. "
                "Either provide a fingerprint_column or include a 'smiles' column in the DataFrame."
            )

        # Fingerprints will be computed in _prepare_data
        return "fingerprint"

    def _prepare_data(self) -> None:
        """Compute fingerprints from SMILES if needed."""
        # If fingerprint column doesn't exist yet, compute it
        if self.fingerprint_column not in self.df.columns:
            log.info(f"Computing Morgan fingerprints (radius={self._fp_radius}, n_bits={self._fp_n_bits})...")
            self.df = compute_morgan_fingerprints(
                self.df, radius=self._fp_radius, n_bits=self._fp_n_bits, counts=self._fp_counts
            )

    def _build_model(self) -> None:
        """
        Build the fingerprint proximity model for Tanimoto similarity.
        Converts fingerprint strings to binary arrays and initializes NearestNeighbors.

        Note: sklearn uses Jaccard distance internally (1 - Tanimoto similarity).
        We convert back to Tanimoto similarity in the output methods.
        """
        log.info("Converting fingerprints to binary feature matrix...")

        # Convert fingerprint strings to binary arrays and store for later use
        self.X = self._fingerprints_to_matrix(self.df)

        # sklearn uses Jaccard distance = 1 - Tanimoto similarity
        # We convert to Tanimoto similarity in neighbors() and _precompute_metrics()
        log.info("Building NearestNeighbors model (Jaccard/Tanimoto metric, BallTree)...")
        self.nn = NearestNeighbors(metric="jaccard", algorithm="ball_tree").fit(self.X)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform fingerprints to binary matrix for querying.

        Args:
            df: DataFrame containing fingerprints to transform.

        Returns:
            Binary feature matrix for the fingerprints.
        """
        return self._fingerprints_to_matrix(df)

    def _fingerprints_to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert fingerprint strings to a binary numpy matrix.

        Args:
            df: DataFrame containing fingerprint column.

        Returns:
            2D numpy array of binary fingerprint bits.
        """
        fingerprint_bits = df[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )
        return np.vstack(fingerprint_bits)

    def _precompute_metrics(self) -> None:
        """Precompute metrics, adding Tanimoto similarity alongside distance."""
        # Call parent to compute nn_distance (Jaccard), nn_id, nn_target, nn_target_diff
        super()._precompute_metrics()

        # Add Tanimoto similarity (keep nn_distance for internal use by target_gradients)
        self.df["nn_similarity"] = 1 - self.df["nn_distance"]

    def _set_core_columns(self) -> None:
        """Set core columns using nn_similarity instead of nn_distance."""
        self.core_columns = [self.id_column, "nn_similarity", "nn_id"]
        if self.target:
            self.core_columns.extend([self.target, "nn_target", "nn_target_diff"])

    def _project_2d(self) -> None:
        """Project the fingerprint matrix to 2D for visualization using UMAP with Jaccard metric."""
        self.df = Projection2D().fit_transform(self.df, feature_matrix=self.X, metric="jaccard")

    def isolated(self, top_percent: float = 1.0) -> pd.DataFrame:
        """
        Find isolated data points based on Tanimoto similarity to nearest neighbor.

        Args:
            top_percent: Percentage of most isolated data points to return (e.g., 1.0 returns top 1%)

        Returns:
            DataFrame of observations with lowest Tanimoto similarity, sorted ascending
        """
        # For Tanimoto similarity, isolated means LOW similarity to nearest neighbor
        percentile = top_percent
        threshold = np.percentile(self.df["nn_similarity"], percentile)
        isolated = self.df[self.df["nn_similarity"] <= threshold].copy()
        isolated = isolated.sort_values("nn_similarity", ascending=True).reset_index(drop=True)
        return isolated if self.include_all_columns else isolated[self.core_columns]

    def proximity_stats(self) -> pd.DataFrame:
        """
        Return distribution statistics for nearest neighbor Tanimoto similarity.

        Returns:
            DataFrame with similarity distribution statistics (count, mean, std, percentiles)
        """
        return (
            self.df["nn_similarity"]
            .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .to_frame()
        )

    def neighbors(
        self,
        id_or_ids: Union[str, int, List[Union[str, int]]],
        n_neighbors: Optional[int] = 5,
        min_similarity: Optional[float] = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """
        Return neighbors for ID(s) from the existing dataset.

        Args:
            id_or_ids: Single ID or list of IDs to look up
            n_neighbors: Number of neighbors to return (default: 5, ignored if min_similarity is set)
            min_similarity: If provided, find all neighbors with Tanimoto similarity >= this value (0-1)
            include_self: Whether to include self in results (default: True)

        Returns:
            DataFrame containing neighbors with Tanimoto similarity scores
        """
        # Convert min_similarity to radius (Jaccard distance = 1 - Tanimoto similarity)
        radius = 1 - min_similarity if min_similarity is not None else None

        # Call parent method (returns Jaccard distance)
        neighbors_df = super().neighbors(
            id_or_ids=id_or_ids,
            n_neighbors=n_neighbors,
            radius=radius,
            include_self=include_self,
        )

        # Convert Jaccard distance to Tanimoto similarity
        neighbors_df["similarity"] = 1 - neighbors_df["distance"]
        neighbors_df.drop(columns=["distance"], inplace=True)

        return neighbors_df


# Testing the FingerprintProximity class
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create an Example DataFrame with fingerprints
    data = {
        "id": ["a", "b", "c", "d", "e"],
        "fingerprint": ["101010", "111010", "101110", "011100", "000111"],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)

    # Test basic FingerprintProximity with explicit fingerprint column
    prox = FingerprintProximity(df, fingerprint_column="fingerprint", id_column="id", target="target")
    print(prox.neighbors("a", n_neighbors=3))

    # Test neighbors with similarity threshold
    print(prox.neighbors("a", min_similarity=0.5))

    # Test with include_all_columns=True
    prox = FingerprintProximity(
        df,
        fingerprint_column="fingerprint",
        id_column="id",
        target="target",
        include_all_columns=True,
    )
    print(prox.neighbors(["a", "b"]))

    # Test on real data from Workbench
    from workbench.api import FeatureSet, Model

    fs = FeatureSet("aqsol_features")
    model = Model("aqsol-regression")
    df = fs.pull_dataframe()
    prox = FingerprintProximity(df, id_column=fs.id_column, target=model.target())

    print("\n" + "=" * 80)
    print("Testing Neighbors...")
    print("=" * 80)
    test_id = df[fs.id_column].tolist()[0]
    print(f"\nNeighbors for ID {test_id}:")
    print(prox.neighbors(test_id))

    print("\n" + "=" * 80)
    print("Testing isolated compounds...")
    print("=" * 80)

    # Test isolated data in the top 1%
    isolated_1pct = prox.isolated(top_percent=1.0)
    print(f"\nTop 1% most isolated compounds (n={len(isolated_1pct)}):")
    print(isolated_1pct)

    # Test isolated data in the top 5%
    isolated_5pct = prox.isolated(top_percent=5.0)
    print(f"\nTop 5% most isolated compounds (n={len(isolated_5pct)}):")
    print(isolated_5pct)

    print("\n" + "=" * 80)
    print("Testing target_gradients...")
    print("=" * 80)

    # Test with different parameters
    gradients_1pct = prox.target_gradients(top_percent=1.0, min_delta=1.0)
    print(f"\nTop 1% target gradients (min_delta=1.0) (n={len(gradients_1pct)}):")
    print(gradients_1pct)

    gradients_5pct = prox.target_gradients(top_percent=5.0, min_delta=5.0)
    print(f"\nTop 5% target gradients (min_delta=5.0) (n={len(gradients_5pct)}):")
    print(gradients_5pct)

    # Test proximity_stats
    print("\n" + "=" * 80)
    print("Testing proximity_stats...")
    print("=" * 80)
    stats = prox.proximity_stats()
    print(stats)

    # Plot the similarity distribution using pandas
    print("\n" + "=" * 80)
    print("Plotting similarity distribution...")
    print("=" * 80)
    prox.df["nn_similarity"].hist(bins=50, figsize=(10, 6), edgecolor="black")

    # Visualize the 2D projection
    print("\n" + "=" * 80)
    print("Visualizing 2D Projection...")
    print("=" * 80)
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    unit_test = PluginUnitTest(ScatterPlot, input_data=prox.df[:1000], x="x", y="y", color=model.target())
    unit_test.run()
