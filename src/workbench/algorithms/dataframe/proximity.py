import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Union, List
import logging

# Set up logging
log = logging.getLogger("workbench")


class Proximity:
    def __init__(
        self,
        df: pd.DataFrame,
        id_column: Union[int, str],
        features: List[str],
        target: str = None,
        n_neighbors: int = 5,
    ) -> None:
        """
        Initialize the Proximity class.

        Args:
            df (pd.DataFrame): DataFrame containing data for neighbor computations.
            id_column (Union[int, str]): Name of the column used as an identifier.
            features (List[str]): List of feature column names to be used for neighbor computations.
            target (str, optional): Name of the target column. Defaults to None.
            n_neighbors (int): Number of neighbors to compute. Defaults to 5.
        """
        self.df = df.dropna(subset=features).copy()
        self.id_column = id_column
        self.n_neighbors = min(n_neighbors, len(self.df) - 1)
        self.target = target
        self.features = features

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Standardize features and fit Nearest Neighbors model."""
        self.X = StandardScaler().fit_transform(self.df[self.features])
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    def all_neighbors(self) -> pd.DataFrame:
        """
        Compute nearest neighbors for all rows in the dataset.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their distances.
        """
        distances, indices = self.nn.kneighbors(self.X)
        results = []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            results.extend(self._build_results(i, dists, nbrs, include_self=False, add_columns=None))
        return pd.DataFrame(results)

    def neighbors(
        self, query_id: Union[int, str], radius: float = None, include_self: bool = True, add_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Return neighbors of the given query ID, either by fixed neighbors or within a radius.

        Args:
            query_id (Union[int, str]): The ID of the query point.
            radius (float, optional): Optional radius within which neighbors are to be included.
            include_self (bool): Whether to include the query ID itself in the neighbor results.
            add_columns (List[str], optional): Optional list of additional columns to include.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their distances.
        """
        if query_id not in self.df[self.id_column].values:
            raise ValueError(f"Query ID {query_id} not found in the DataFrame")
        query_idx = self.df.index[self.df[self.id_column] == query_id][0]
        if radius is not None:
            distances, indices = self.nn.radius_neighbors([self.X[query_idx]], radius=radius)
            distances, indices = distances[0], indices[0]
        else:
            distances, indices = self.nn.kneighbors([self.X[query_idx]])
            distances, indices = distances[0], indices[0]
        return pd.DataFrame(self._build_results(query_idx, distances, indices, include_self, add_columns))

    def _build_results(
        self, query_idx: int, distances, indices, include_self: bool, add_columns: List[str]
    ) -> List[dict]:
        """Internal: Convert indices and distances to a list of dictionaries."""
        results = []
        query_id = self.df.at[query_idx, self.id_column]
        for neighbor_idx, dist in zip(indices, distances):
            if not include_self and query_idx == neighbor_idx:
                continue
            neighbor_info = {
                self.id_column: query_id,
                "neighbor_id": self.df.at[neighbor_idx, self.id_column],
                "distance": dist,
            }
            # Collect extra columns if they exist.
            relevant_cols = (
                [self.target, "prediction"]
                + [c for c in self.df.columns if "_proba" in c or "residual" in c]
                + ["outlier"]
                + (add_columns or [])
            )
            for col in filter(lambda c: c in self.df.columns, relevant_cols):
                neighbor_info[col] = self.df.iloc[neighbor_idx][col]
            results.append(neighbor_info)
        return results


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
    prox = Proximity(df, id_column="ID", features=features, n_neighbors=3)
    print(prox.all_neighbors())

    # Test the neighbors method
    print(prox.neighbors(query_id=1))

    # Test the neighbors method with radius
    print(prox.neighbors(query_id=1, radius=2.0))

    # Test with Features list
    prox = Proximity(df, id_column="ID", features=["Feature1"], n_neighbors=2)
    print(prox.all_neighbors())

    # Create a sample DataFrame
    data = {
        "ID": ["a", "b", "c", "d", "e"],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)

    # Test with String Ids
    prox = Proximity(df, id_column="ID", features=["Feature1", "Feature2"], target="target", n_neighbors=3)
    print(prox.all_neighbors())

    # Test the neighbors method
    print(prox.neighbors(query_id="a", add_columns=["Feature1", "Feature2"]))
