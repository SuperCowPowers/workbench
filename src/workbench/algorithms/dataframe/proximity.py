import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union, List


class Proximity:
    def __init__(self, df: pd.DataFrame, id_column: Union[int, str], features: List[str] = None, n_neighbors: int = 10) -> None:
        """
        Initialize the Proximity class.

        Args:
            df (pd.DataFrame): DataFrame containing data for neighbor computations.
            id_column (Union[int, str]): Name of the column used as an identifier.
            features (List[str]): List of feature column names to be used for neighbor computations.
            n_neighbors (int): Number of neighbors to compute.
        """
        self.df = df.copy()
        self.id_column = id_column
        self.n_neighbors = n_neighbors

        # Automatically determine features if not provided
        self.features = features or self._auto_features()
        self._prepare_data()

    def _auto_features(self) -> List[str]:
        """Automatically determine feature columns, excluding the ID column."""
        return self.df.select_dtypes(include=[np.number]).columns.difference([self.id_column]).tolist()

    def _prepare_data(self) -> None:
        """Prepare the data for nearest neighbor computation."""
        self.X = self.df[self.features].values
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    def _prepare_data(self) -> None:
        if not self.features:
            # Default to all numeric columns except the ID column
            self.features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.features.remove(self.id_column)

        self.X = self.df[self.features].values
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    def all_neighbors(self, include_self: bool = False) -> pd.DataFrame:
        """
        Compute nearest neighbors for all rows in the dataset.

        Args:
            include_self (bool): Whether to include self-loops in the results.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their distances.
        """
        results = self._get_neighbors(query_idx=None, include_self=include_self)
        return pd.DataFrame(results)

    def neighbors(
        self, query_id: Union[int, str], radius: float = None, include_self: bool = False
    ) -> pd.DataFrame:
        """
        Return neighbors of the given query ID, either by fixed neighbors or within a radius.

        Args:
            query_id (Union[int, str]): The ID of the query point.
            radius (float): Optional radius within which neighbors are to be included.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their distances.
        """
        query_idx = self.df.index[self.df[self.id_column] == query_id].tolist()
        if not query_idx:
            raise ValueError(f"Query ID {query_id} not found in the DataFrame")
        query_idx = query_idx[0]

        results = self._get_neighbors(query_idx=query_idx, radius=radius, include_self=include_self)
        return pd.DataFrame(results)

    def _get_neighbors(self, query_idx: int = None, radius: float = None, include_self: bool = True) -> List[dict]:
        """
        Internal: Helper method to compute neighbors for a given query index or all rows.

        Args:
            query_idx (int): Index of the query point. If None, computes for all rows.
            radius (float): Optional radius threshold.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            List[dict]: List of dictionaries with neighbor information.
        """
        if query_idx is None:  # Handle all_neighbors case
            distances, indices = self.nn.kneighbors(self.X)
        elif radius is not None:  # Radius-based search
            distances, indices = self.nn.radius_neighbors([self.X[query_idx]], radius=radius)
        else:  # Fixed neighbors for a single query
            distances, indices = self.nn.kneighbors([self.X[query_idx]])

        distances, indices = np.array(distances), np.array(indices)

        results = []
        if query_idx is None:
            for idx, (neighbors, dists) in enumerate(zip(indices, distances)):
                for neighbor_idx, dist in zip(neighbors, dists):
                    if not include_self and idx == neighbor_idx:
                        continue
                    results.append({
                        self.id_column: self.df.iloc[idx][self.id_column],
                        "neighbor_id": self.df.iloc[neighbor_idx][self.id_column],
                        "distance": dist,
                    })
        else:
            for neighbor_idx, dist in zip(indices[0], distances[0]):
                if not include_self and neighbor_idx == query_idx:
                    continue
                results.append({
                    self.id_column: self.df.iloc[query_idx][self.id_column],
                    "neighbor_id": self.df.iloc[neighbor_idx][self.id_column],
                    "distance": dist,
                })

        return results


# Testing the Proxmity class
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create a sample DataFrame
    data = {
        "ID": [1, 2, 3, 4, 5],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
    }
    df = pd.DataFrame(data)

    # Test the Proximity class
    prox = Proximity(df, id_column="ID", n_neighbors=3)
    print(prox.all_neighbors())

    # Test the neighbors method
    print(prox.neighbors(query_id=1))

    # Test the neighbors method with radius
    print(prox.neighbors(query_id=1, radius=0.3))

    # Test with Features list
    prox = Proximity(df, id_column="ID", features=["Feature1"], n_neighbors=2)
    print(prox.all_neighbors())


    # Create a sample DataFrame
    data = {
        "ID": ["a", "b", "c", "d", "e"],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
    }
    df = pd.DataFrame(data)

    # Test with String Ids
    prox = Proximity(df, id_column="ID", n_neighbors=3)
    print(prox.all_neighbors())

