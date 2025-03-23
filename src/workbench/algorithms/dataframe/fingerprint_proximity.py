import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union, List
import logging

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity, ProximityType

# Set up logging
log = logging.getLogger("workbench")


class FingerprintProximity(Proximity):
    def __init__(
        self, df: pd.DataFrame, id_column: Union[int, str], fingerprint_column: str, n_neighbors: int = 5
    ) -> None:
        """
        Initialize the FingerprintProximity class for binary fingerprint similarity.

        Args:
            df (pd.DataFrame): DataFrame containing fingerprints.
            id_column (Union[int, str]): Name of the column used as an identifier.
            fingerprint_column (str): Name of the column containing fingerprints.
            n_neighbors (int): Default number of neighbors to compute.
        """
        self.fingerprint_column = fingerprint_column

        # Call the parent class constructor
        super().__init__(df, id_column=id_column, features=[fingerprint_column], n_neighbors=n_neighbors)

    # Override the build_proximity_model method
    def build_proximity_model(self) -> None:
        """
        Prepare the fingerprint data for nearest neighbor calculations.
        Converts fingerprint strings to binary arrays and initializes NearestNeighbors.
        """
        log.info("Converting fingerprints to binary feature matrix...")
        self.proximity_type = ProximityType.SIMILARITY

        # Convert fingerprint strings to binary arrays

        fingerprint_bits = self.df[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )
        self.X = np.vstack(fingerprint_bits)

        # Use Jaccard similarity for binary fingerprints
        log.info("Computing NearestNeighbors with Jaccard metric...")
        self.nn = NearestNeighbors(metric="jaccard", n_neighbors=self.n_neighbors + 1).fit(self.X)

    # Override the prep_features_for_query method
    def prep_features_for_query(self, query_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare the query DataFrame by converting fingerprints to binary arrays.

        Args:
            query_df (pd.DataFrame): DataFrame containing query fingerprints.

        Returns:
            np.ndarray: Binary feature matrix for the query fingerprints.
        """
        fingerprint_bits = query_df[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )
        return np.vstack(fingerprint_bits)

    def all_neighbors(
        self,
        min_similarity: float = None,
        include_self: bool = False,
        add_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Find neighbors for all fingerprints in the dataset.

        Args:
            min_similarity: Minimum similarity threshold (0-1)
            include_self: Whether to include self in results
            add_columns: Additional columns to include in results

        Returns:
            DataFrame containing neighbors and similarities
        """

        # Call the parent class method to find neighbors
        return self.neighbors(
            query_df=self.df,
            min_similarity=min_similarity,
            include_self=include_self,
            add_columns=add_columns,
        )

    def neighbors(
        self,
        query_df: pd.DataFrame,
        min_similarity: float = None,
        include_self: bool = False,
        add_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Find neighbors for each row in the query DataFrame.

        Args:
            query_df: DataFrame containing query fingerprints
            min_similarity: Minimum similarity threshold (0-1)
            include_self: Whether to include self in results (if present)
            add_columns: Additional columns to include in results

        Returns:
            DataFrame containing neighbors and similarities

        Note: The query DataFrame must include the feature columns. The id_column is optional.
        """

        # Calculate radius from similarity if provided
        radius = 1 - min_similarity if min_similarity is not None else None

        # Call the parent class method to find neighbors
        neighbors_df = super().neighbors(
            query_df=query_df,
            radius=radius,
            include_self=include_self,
            add_columns=add_columns,
        )

        # Convert distances to similarity
        neighbors_df["similarity"] = 1 - neighbors_df["distance"]
        neighbors_df.drop(columns=["distance"], inplace=True)
        return neighbors_df


# Testing the FingerprintProximity class
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Example DataFrame
    data = {
        "id": ["a", "b", "c", "d"],
        "fingerprint": ["101010", "111010", "101110", "011100"],
    }
    df = pd.DataFrame(data)

    # Initialize the FingerprintProximity class
    proximity = FingerprintProximity(df, fingerprint_column="fingerprint", id_column="id", n_neighbors=3)

    # Test 1: All neighbors
    print("\n--- Test 1: All Neighbors ---")
    all_neighbors_df = proximity.all_neighbors()
    print(all_neighbors_df)

    # Test 2: Neighbors for a specific query
    print("\n--- Test 2: Neighbors for Query ---")
    query_df = pd.DataFrame({"id": ["a"], "fingerprint": ["101010"]})
    query_neighbors_df = proximity.neighbors(query_df=query_df)
    print(query_neighbors_df)

    # Test 3: Neighbors with similarity threshold
    print("\n--- Test 3: Neighbors with Minimum Similarity 0.5 ---")
    query_neighbors_sim_df = proximity.neighbors(query_df=query_df, min_similarity=0.5)
    print(query_neighbors_sim_df)
