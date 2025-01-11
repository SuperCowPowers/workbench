"""FingerprintProximity: A class for neighbor lookups using KNN on fingerprints."""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union, List

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity


class FingerprintProximity(Proximity):
    def __init__(self, df: pd.DataFrame, fingerprint_column: str, id_column: str, n_neighbors: int = 10) -> None:
        """
        Initialize the FingerprintProximity class.

        Args:
            df (pd.DataFrame): DataFrame containing fingerprints and other features.
            fingerprint_column (str): Name of the column containing fingerprints.
            id_column (str): Name of the column used as an identifier.
            neighbors (int): Number of neighbors to compute.
        """
        self.df = df.copy()
        self.fingerprint_column = fingerprint_column
        self.id_column = id_column
        self.n_neighbors = n_neighbors
        self._prepare_data()

    def _prepare_data(self) -> None:
        """
        Prepare the DataFrame by converting fingerprints and setting up the feature matrix.
        """

        # Convert the bitstring fingerprint into a NumPy array of integers
        self.df["fingerprint_bits"] = self.df[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )

        # Create a matrix of fingerprints
        self.X = np.vstack(self.df["fingerprint_bits"].values)

        # Fit NearestNeighbors
        self.nn = NearestNeighbors(metric="jaccard", n_neighbors=self.n_neighbors).fit(self.X)

    def all_neighbors(self, include_self: bool = False) -> pd.DataFrame:
        """
        Compute nearest neighbors for all rows in the dataset.

        Args:
            include_self (bool): Whether to include self-loops in the results.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their Tanimoto similarities.
        """
        results = self._get_neighbors(query_idx=None, include_self=include_self)
        return pd.DataFrame(results)

    def neighbors(
        self, query_id: Union[str, int], similarity: float = None, include_self: bool = False
    ) -> pd.DataFrame:
        """
        Return neighbors of the given query ID, either by fixed neighbors or above a similarity threshold.

        Args:
            query_id (Union[str, int]): The ID of the query point.
            similarity (float): Optional similarity threshold above which neighbors are to be included.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            pd.DataFrame: Filtered DataFrame that includes the query ID, its neighbors, and their similarities.
        """
        # Find the query point index
        query_idx = self.df.index[self.df[self.id_column] == query_id].tolist()
        if not query_idx:
            raise ValueError(f"Query ID {query_id} not found in the DataFrame")
        query_idx = query_idx[0]

        results = self._get_neighbors(query_idx=query_idx, similarity=similarity, include_self=include_self)
        return pd.DataFrame(results)

    def _get_neighbors(self, query_idx: int = None, similarity: float = None, include_self: bool = True) -> List[dict]:
        """
        Internal: Helper method to compute neighbors for a given query index or all rows.

        Args:
            query_idx (int): Index of the query point. If None, computes for all rows.
            similarity (float): Optional similarity threshold above which neighbors are to be included.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            List[dict]: List of dictionaries with neighbor information.
        """
        if query_idx is None:  # Handle all_neighbors case
            distances, indices = self.nn.kneighbors(self.X)
        elif similarity is not None:  # Radius-based search
            radius = 1 - similarity
            distances, indices = self.nn.radius_neighbors([self.X[query_idx]], radius=radius)
        else:  # Fixed neighbors for a single query
            distances, indices = self.nn.kneighbors([self.X[query_idx]])

        # Normalize data structure
        distances, indices = np.array(distances), np.array(indices)

        # Build results
        results = []
        if query_idx is None:  # Compute for all rows
            for idx, (neighbors, dists) in enumerate(zip(indices, distances)):
                for neighbor_idx, dist in zip(neighbors, dists):
                    if not include_self and idx == neighbor_idx:
                        continue
                    results.append(
                        {
                            self.id_column: self.df.iloc[idx][self.id_column],
                            "neighbor_id": self.df.iloc[neighbor_idx][self.id_column],
                            "similarity": 1 - dist,
                        }
                    )
        else:  # Single query
            for neighbor_idx, dist in zip(indices[0], distances[0]):
                if not include_self and neighbor_idx == query_idx:
                    continue
                results.append(
                    {
                        self.id_column: self.df.iloc[query_idx][self.id_column],
                        "neighbor_id": self.df.iloc[neighbor_idx][self.id_column],
                        "similarity": 1 - dist,
                    }
                )

        return results


# Testing the FeatureSpaceProximity class with separate training and test/evaluation dataframes
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Example DataFrame
    data = {
        "id": [1, 2, 3, 4],
        "fingerprint": ["101010", "111010", "101110", "011100"],
    }
    df = pd.DataFrame(data)

    # Initialize and compute
    proximity = FingerprintProximity(df, fingerprint_column="fingerprint", id_column="id", n_neighbors=4)

    # Get all neighbors
    all_neighbors_df = proximity.all_neighbors()
    print(all_neighbors_df)

    # Get neighbors for a specific query
    query_neighbors_df = proximity.neighbors(query_id=1)
    print(query_neighbors_df)

    # Get neighbors for a specific query
    query_neighbors_df = proximity.neighbors(query_id=1, similarity=0.5)
    print(query_neighbors_df)
