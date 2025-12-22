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
    """Proximity computations for binary fingerprints using Jaccard/Tanimoto similarity."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        fingerprint_column: Optional[str] = None,
        target: Optional[str] = None,
        track_columns: Optional[List[str]] = None,
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
            track_columns: Additional columns to track in results. Defaults to None.
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
            track_columns=track_columns,
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
        Build the fingerprint proximity model using Jaccard metric.
        Converts fingerprint strings to binary arrays and initializes NearestNeighbors.
        """
        log.info("Converting fingerprints to binary feature matrix...")

        # Convert fingerprint strings to binary arrays and store for later use
        self.X = self._fingerprints_to_matrix(self.df)

        # Use Jaccard distance for binary fingerprints (1 - Tanimoto similarity)
        # Using BallTree algorithm for better performance with high-dimensional binary data
        log.info("Computing NearestNeighbors with Jaccard metric (BallTree)...")
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

    def _project_2d(self) -> None:
        """Project the fingerprint matrix to 2D for visualization using UMAP with Jaccard metric."""
        self.df = Projection2D().fit_transform(self.df, feature_matrix=self.X, metric="jaccard")

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
            min_similarity: If provided, find all neighbors with similarity >= this value (0-1)
            include_self: Whether to include self in results (default: True)

        Returns:
            DataFrame containing neighbors with similarity scores (1 - Jaccard distance)
        """
        # Convert min_similarity to radius (Jaccard distance)
        radius = 1 - min_similarity if min_similarity is not None else None

        # Call parent method
        neighbors_df = super().neighbors(
            id_or_ids=id_or_ids,
            n_neighbors=n_neighbors,
            radius=radius,
            include_self=include_self,
        )

        # Convert distance to similarity for fingerprints
        neighbors_df["similarity"] = 1 - neighbors_df["distance"]
        neighbors_df.drop(columns=["distance"], inplace=True)

        return neighbors_df


# Testing the FingerprintProximity class
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create an Example DataFrame
    data = {
        "id": ["a", "b", "c", "d", "e"],
        "fingerprint": ["101010", "111010", "101110", "011100", "000111"],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)

    # Initialize the FingerprintProximity class
    proximity = FingerprintProximity(df, fingerprint_column="fingerprint", id_column="id", target="target")

    # Test 1: Neighbors for a single ID
    print("\n--- Test 1: Neighbors for ID 'a' ---")
    neighbors_df = proximity.neighbors("a")
    print(neighbors_df)

    # Test 2: Neighbors for multiple IDs
    print("\n--- Test 2: Neighbors for IDs ['a', 'b'] ---")
    neighbors_df = proximity.neighbors(["a", "b"])
    print(neighbors_df)

    # Test 3: Neighbors with similarity threshold
    print("\n--- Test 3: Neighbors with min_similarity=0.5 ---")
    neighbors_df = proximity.neighbors("a", min_similarity=0.5)
    print(neighbors_df)

    # Test 4: Isolated compounds
    print("\n--- Test 4: Isolated compounds (top 50%) ---")
    isolated_df = proximity.isolated(top_percent=50.0)
    print(isolated_df[["id", "nn_distance", "nn_id"]])

    # Test 5: Target gradients
    print("\n--- Test 5: Target gradients ---")
    gradients_df = proximity.target_gradients(top_percent=50.0, min_delta=0.1)
    print(gradients_df)

    # Test on real data from Workbench
    from workbench.api import FeatureSet, Model

    fs = FeatureSet("aqsol_features")
    model = Model("aqsol-regression")
    features = model.features()
    df = fs.pull_dataframe()
    prox = FingerprintProximity(df, id_column=fs.id_column, target=model.target())
    print(prox.neighbors(df[fs.id_column].tolist()[:3]))

    # Test 6: Visualize the 2D projection
    print("\n--- Test 6: 2D Projection Scatter Plot ---")
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    unit_test = PluginUnitTest(ScatterPlot, input_data=prox.df[:1000], x="x", y="y")
    unit_test.run()
