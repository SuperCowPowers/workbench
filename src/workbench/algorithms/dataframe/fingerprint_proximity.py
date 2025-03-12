import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union, List, Dict, Optional
import logging

# Set up logging
log = logging.getLogger("workbench")


class FingerprintProximity:
    def __init__(
            self,
            df: pd.DataFrame,
            fingerprint_column: str,
            id_column: Union[int, str],
            n_neighbors: int = 10
    ) -> None:
        """
        Initialize the FingerprintProximity class for binary fingerprint similarity.

        Args:
            df (pd.DataFrame): DataFrame containing fingerprints.
            fingerprint_column (str): Name of the column containing fingerprints.
            id_column (Union[int, str]): Name of the column used as an identifier.
            n_neighbors (int): Default number of neighbors to compute.
        """
        self.df = df.copy()
        self.fingerprint_column = fingerprint_column
        self.id_column = id_column
        self.n_neighbors = min(n_neighbors, len(self.df) - 1)

        # Prepare the data
        self._prepare_fingerprint_data()

    def _prepare_fingerprint_data(self) -> None:
        """
        Prepare the fingerprint data for nearest neighbor calculations.
        Converts fingerprint strings to binary arrays and initializes NearestNeighbors.
        """
        log.info("Converting fingerprints to binary feature matrix...")

        # Filter out rows with missing fingerprints
        self.df = self.df.dropna(subset=[self.fingerprint_column])

        # Convert fingerprint strings to binary arrays
        fingerprint_bits = self.df[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )
        self.X = np.vstack(fingerprint_bits)

        # Use Jaccard similarity for binary fingerprints
        log.info("Computing NearestNeighbors with Jaccard metric...")
        self.nn = NearestNeighbors(
            metric="jaccard",
            n_neighbors=self.n_neighbors + 1
        ).fit(self.X)

    def _build_neighbor_result(
            self,
            query_id,
            neighbor_idx: int,
            similarity: float,
            add_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Build a result dictionary for a single neighbor.

        Args:
            query_id: ID of the query point
            neighbor_idx: Index of the neighbor in the original DataFrame
            similarity: Similarity between query and neighbor
            add_columns: Additional columns to include in result

        Returns:
            Dictionary containing neighbor information
        """
        neighbor_id = self.df.iloc[neighbor_idx][self.id_column]

        # Basic neighbor info
        neighbor_info = {
            self.id_column: query_id,
            "neighbor_id": neighbor_id,
            "similarity": similarity,
        }

        # Add user-specified additional columns
        if add_columns:
            for col in filter(lambda c: c in self.df.columns, add_columns):
                neighbor_info[col] = self.df.iloc[neighbor_idx][col]

        return neighbor_info

    def neighbors(
            self,
            query_df: pd.DataFrame,
            min_similarity: float = None,
            max_neighbors: int = None,
            include_self: bool = False,
            add_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Find neighbors for each row in the query DataFrame.

        Args:
            query_df: DataFrame containing query fingerprints
            min_similarity: Minimum similarity threshold (0-1)
            max_neighbors: Maximum number of neighbors to return per query
            include_self: Whether to include self in results (if present)
            add_columns: Additional columns to include in results

        Returns:
            DataFrame containing neighbors and similarities

        Note: The query DataFrame must include both the fingerprint_column and id_column.
        """
        # Verify required columns are present
        required_cols = {self.fingerprint_column, self.id_column}
        missing = required_cols - set(query_df.columns)
        if missing:
            raise ValueError(f"Query DataFrame is missing required columns: {missing}")

        # Determine neighbors to find
        n_neighbors = max_neighbors if max_neighbors is not None else self.n_neighbors
        n_neighbors = min(n_neighbors + 1, len(self.df))  # +1 to account for self

        # Convert fingerprints to binary arrays
        fingerprint_bits = query_df[self.fingerprint_column].apply(
            lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
        )
        X_query = np.vstack(fingerprint_bits)

        # Calculate radius from similarity if provided
        radius = 1 - min_similarity if min_similarity is not None else None

        # Find neighbors based on radius or k-nearest
        if radius is not None:
            distances, indices = self.nn.radius_neighbors(X_query, radius=radius)
        else:
            distances, indices = self.nn.kneighbors(X_query, n_neighbors=n_neighbors)

        # Build results
        all_results = []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            query_id = query_df.iloc[i][self.id_column]

            for neighbor_idx, dist in zip(nbrs, dists):
                # Get neighbor ID
                neighbor_id = self.df.iloc[neighbor_idx][self.id_column]

                # Skip if the neighbor is the query itself and include_self is False
                if not include_self and neighbor_id == query_id:
                    continue

                # Convert distance to similarity
                similarity = 1 - dist

                all_results.append(self._build_neighbor_result(
                    query_id=query_id,
                    neighbor_idx=neighbor_idx,
                    similarity=similarity,
                    add_columns=add_columns
                ))

        return pd.DataFrame(all_results)

    def all_neighbors(
            self,
            min_similarity: float = None,
            max_neighbors: int = None,
            include_self: bool = False,
            add_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Find neighbors for all fingerprints in the dataset.

        Args:
            min_similarity: Minimum similarity threshold (0-1)
            max_neighbors: Maximum number of neighbors to return per fingerprint
            include_self: Whether to include self in results
            add_columns: Additional columns to include in results

        Returns:
            DataFrame containing neighbors and similarities
        """
        # Use the same method but with the original dataframe as query
        return self.neighbors(
            query_df=self.df,
            min_similarity=min_similarity,
            max_neighbors=max_neighbors,
            include_self=include_self,
            add_columns=add_columns
        )


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