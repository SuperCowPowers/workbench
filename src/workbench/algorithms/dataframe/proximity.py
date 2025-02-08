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
        features: List[str] = None,
        target: str = None,
        n_neighbors: int = 5,
    ) -> None:
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
        self.target = target

        # Automatically determine features if not provided
        self.features = features or self._auto_features()

        # Check for NaNs within the features and if we so we drop them here
        orig_len = len(self.df)
        self.df = self.df.dropna(subset=self.features)
        if len(self.df) < orig_len:
            log.warning(f"Dropped {orig_len - len(self.df)} rows with NaNs in the feature columns")

        # Call the internal method to prepare the data (overridden in subclasses)
        self._prepare_data()

        # Store the min and max distances for normalization
        self.min_distance = None
        self.max_distance = None

    def _auto_features(self) -> List[str]:
        """Automatically determine feature columns, excluding the ID column and target column."""
        return self.df.select_dtypes(include=[np.number]).columns.difference([self.id_column, self.target]).tolist()

    def _prepare_data(self) -> None:
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.df[self.features].values)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    def get_edge_weight(self, row: pd.Series) -> float:
        """

        Args:
            row (pd.Series): A row from the all_neighbors DataFrame.

        Returns:
            float: The computed edge weight.
        """
        # Normalized distance-based weight
        return 1.0 - (row["distance"] - self.min_distance) / (self.max_distance - self.min_distance)

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
        self, query_id: Union[int, str], radius: float = None, include_self: bool = True, add_columns: list = None
    ) -> pd.DataFrame:
        """
        Return neighbors of the given query ID, either by fixed neighbors or within a radius.

        Args:
            query_id (Union[int, str]): The ID of the query point.
            radius (float): Optional radius within which neighbors are to be included.
            include_self (bool): Whether to include the query ID itself in the neighbor results.
            add_columns (list): Optional list of additional columns to include in the results.

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their distances.
        """
        # Map the query_id to its positional index in self.df
        query_idx = self.df.index[self.df[self.id_column] == query_id].tolist()
        if not query_idx:
            raise ValueError(f"Query ID {query_id} not found in the DataFrame")

        # Use the positional index to query self.X
        query_idx = self.df.index.get_loc(query_idx[0])

        # Compute neighbors
        results = self._get_neighbors(
            query_idx=query_idx, radius=radius, include_self=include_self, add_columns=add_columns
        )
        return pd.DataFrame(results)

    def _get_neighbors(
        self, query_idx: int = None, radius: float = None, include_self: bool = True, add_columns: list = None
    ) -> List[dict]:
        """
        Internal: Helper method to compute neighbors for a given query index or all rows.

        Args:
            query_idx (int, optional): Index of the query point. If None, computes for all rows.
            radius (float, optional): Optional radius threshold.
            include_self (bool): Whether to include the query ID itself in the neighbor results.
            add_columns (list): Optional list of additional columns to include in the results.

        Returns:
            List[dict]: List of dictionaries with neighbor information.
        """
        if query_idx is None:
            log.info(f"Computing NearestNeighbors with input size: {self.X.shape}...")
            distances, indices = self.nn.kneighbors(self.X)
        else:
            query_point = [self.X[query_idx]]
            distances, indices = (
                self.nn.radius_neighbors(query_point, radius=radius)
                if radius is not None
                else self.nn.kneighbors(query_point)
            )

        self.min_distance, self.max_distance = distances.min(), distances.max()

        results = []
        query_indices = range(len(self.X)) if query_idx is None else [query_idx]

        for i, (neighbors, dists) in zip(query_indices, zip(indices, distances)):
            query_id = self.df.iloc[i][self.id_column]
            for neighbor_idx, dist in zip(neighbors, dists):
                if not include_self and i == neighbor_idx:
                    continue
                neighbor_info = {
                    self.id_column: query_id,
                    "neighbor_id": self.df.iloc[neighbor_idx][self.id_column],
                    "distance": dist,
                }

                # Optionally include the target column
                if self.target:
                    neighbor_info[self.target] = self.df.iloc[neighbor_idx][self.target]

                # Check for any prediction, residual, or _proba columns
                if "prediction" in self.df.columns:
                    neighbor_info["prediction"] = self.df.iloc[neighbor_idx]["prediction"]
                for col in self.df.columns:
                    if "_proba" in col or "residual" in col:
                        neighbor_info[col] = self.df.iloc[neighbor_idx][col]

                # Optionally include additional columns
                if add_columns:
                    for col in add_columns:
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
        "target": [1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)

    # Test with String Ids
    prox = Proximity(df, id_column="ID", target="target", n_neighbors=3)
    print(prox.all_neighbors())

    # Test the neighbors method
    print(prox.neighbors(query_id="a", add_columns=["Feature1", "Feature2"]))
