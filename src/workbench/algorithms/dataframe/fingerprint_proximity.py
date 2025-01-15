import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union
from workbench.algorithms.dataframe.proximity import Proximity
import logging

# Set up logging
log = logging.getLogger("workbench")


class FingerprintProximity(Proximity):
    def __init__(
        self, df: pd.DataFrame, fingerprint_column: str, id_column: Union[int, str], n_neighbors: int = 10
    ) -> None:
        """
        Initialize the FingerprintProximity class.

        Args:
            df (pd.DataFrame): DataFrame containing fingerprints and other features.
            fingerprint_column (str): Name of the column containing fingerprints.
            id_column (Union[int, str]): Name of the column used as an identifier.
            n_neighbors (int): Number of neighbors to compute.
        """
        self.fingerprint_column = fingerprint_column
        super().__init__(df, id_column=id_column, n_neighbors=n_neighbors)

    def _prepare_data(self) -> None:
        """
        Prepare the DataFrame by converting fingerprints into a binary feature matrix.
        """
        # Convert the fingerprint strings to binary arrays
        log.info("Converting fingerprints to binary feature matrix...")
        fingerprint_bits = self.data[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )
        self.X = np.vstack(fingerprint_bits)

        # Use Jaccard similarity for binary fingerprints
        log.info("Computing NearestNeighbors with Jaccard metric...")
        self.nn = NearestNeighbors(metric="jaccard", n_neighbors=self.n_neighbors + 1).fit(self.X)

    def get_edge_weight(self, row: pd.Series) -> float:
        """
        Compute edge weight using similarity for fingerprints.
        """
        return row["similarity"]

    def neighbors(
        self, query_id: Union[int, str], similarity: float = None, include_self: bool = False
    ) -> pd.DataFrame:
        """
        Return neighbors of the given query ID, either by fixed neighbors or above a similarity threshold.

        Args:
            query_id (Union[int, str]): The ID of the query point.
            similarity (float): Optional similarity threshold above which neighbors are to be included.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            pd.DataFrame: Filtered DataFrame that includes the query ID, its neighbors, and their similarities.
        """
        # Convert similarity to a radius (1 - similarity)
        radius = 1 - similarity if similarity is not None else None
        neighbors_df = super().neighbors(query_id=query_id, radius=radius, include_self=include_self)

        # Convert distances to Tanimoto similarities
        if "distance" in neighbors_df.columns:
            neighbors_df["similarity"] = 1 - neighbors_df["distance"]
            neighbors_df = neighbors_df.drop(columns=["distance"])

        return neighbors_df

    def all_neighbors(self, include_self: bool = False) -> pd.DataFrame:
        """
        Compute nearest neighbors for all rows in the dataset.

        Args:
            include_self (bool): Whether to include self-loops in the results.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their Tanimoto similarities.
        """
        all_neighbors_df = super().all_neighbors(include_self=include_self)

        # Convert distances to Tanimoto similarities
        if "distance" in all_neighbors_df.columns:
            all_neighbors_df["similarity"] = 1 - all_neighbors_df["distance"]
            all_neighbors_df = all_neighbors_df.drop(columns=["distance"])

        return all_neighbors_df


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
    print("\n--- Test 2: Neighbors for Query ID 1 ---")
    query_neighbors_df = proximity.neighbors(query_id="a")
    print(query_neighbors_df)

    # Test 3: Neighbors for a specific query with similarity threshold
    print("\n--- Test 3: Neighbors for Query ID 1 with Similarity 0.5 ---")
    query_neighbors_sim_df = proximity.neighbors(query_id="a", similarity=0.5)
    print(query_neighbors_sim_df)
