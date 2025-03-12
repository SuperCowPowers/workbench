import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Union, List, Dict, Tuple, Optional
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
        self.scaler = StandardScaler()

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Standardize features and fit Nearest Neighbors model."""
        self.X = self.scaler.fit_transform(self.df[self.features])
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    def all_neighbors(self, add_columns: List[str] = None) -> pd.DataFrame:
        """
        Compute nearest neighbors for all rows in the dataset.

        Args:
            add_columns: Additional columns to include in results

        Returns:
            pd.DataFrame: A DataFrame of neighbors and their distances.
        """
        distances, indices = self.nn.kneighbors(self.X)
        results = []

        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            query_id = self.df.iloc[i][self.id_column]

            # Process neighbors
            for neighbor_idx, dist in zip(nbrs, dists):
                # Skip self (neighbor index == current row index)
                if neighbor_idx == i:
                    continue
                results.append(self._build_neighbor_result(
                    query_id=query_id,
                    neighbor_idx=neighbor_idx,
                    distance=dist,
                    add_columns=add_columns
                ))

        return pd.DataFrame(results)

    def neighbors(
            self,
            query_df: pd.DataFrame,
            radius: float = None,
            include_self: bool = True,
            add_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Return neighbors for rows in a query DataFrame.

        Args:
            query_df: DataFrame containing query points
            radius: If provided, find all neighbors within this radius
            include_self: Whether to include self in results (if present)
            add_columns: Additional columns to include in results

        Returns:
            DataFrame containing neighbors and distances

        Note: The query DataFrame must include the feature columns and the id_column.
        """
        # Verify required columns are present
        required_cols = set(self.features + [self.id_column])
        missing = required_cols - set(query_df.columns)
        if missing:
            raise ValueError(f"Query DataFrame is missing required columns: {missing}")

        # Transform the query features using the model's scaler
        X_query = self.scaler.transform(query_df[self.features])

        # Get neighbors using either radius or k-nearest neighbors
        if radius is not None:
            distances, indices = self.nn.radius_neighbors(X_query, radius=radius)
        else:
            distances, indices = self.nn.kneighbors(X_query)

        # Build results
        all_results = []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            query_id = query_df.iloc[i][self.id_column]

            for neighbor_idx, dist in zip(nbrs, dists):
                # Skip if the neighbor is the query itself and include_self is False
                neighbor_id = self.df.iloc[neighbor_idx][self.id_column]
                if not include_self and neighbor_id == query_id:
                    continue

                all_results.append(self._build_neighbor_result(
                    query_id=query_id,
                    neighbor_idx=neighbor_idx,
                    distance=dist,
                    add_columns=add_columns
                ))

        return pd.DataFrame(all_results)

    def _build_neighbor_result(
            self,
            query_id,
            neighbor_idx: int,
            distance: float,
            add_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Internal: Build a result dictionary for a single neighbor.

        Args:
            query_id: ID of the query point
            neighbor_idx: Index of the neighbor in the original DataFrame
            distance: Distance between query and neighbor
            add_columns: Additional columns to include in result

        Returns:
            Dictionary containing neighbor information
        """
        neighbor_id = self.df.iloc[neighbor_idx][self.id_column]

        # Basic neighbor info
        neighbor_info = {
            self.id_column: query_id,
            "neighbor_id": neighbor_id,
            "distance": distance,
        }

        # Determine which additional columns to include
        relevant_cols = [self.target, "prediction"] if self.target else []
        relevant_cols += [c for c in self.df.columns if "_proba" in c or "residual" in c]
        relevant_cols += ["outlier"]

        # Add user-specified columns
        if add_columns:
            relevant_cols += add_columns

        # Add values for each relevant column that exists in the dataframe
        for col in filter(lambda c: c in self.df.columns, relevant_cols):
            neighbor_info[col] = self.df.iloc[neighbor_idx][col]

        return neighbor_info


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
    print(prox.neighbors(query_df=df.iloc[[0]]))

    # Test the neighbors method with radius
    print(prox.neighbors(query_df=df.iloc[0:2], radius=2.0))

    # Test with data that isn't in the 'train' dataframe
    query_data = {
        "ID": [6],
        "Feature1": [0.31],
        "Feature2": [0.31],
        "Feature3": [2.31],
    }
    query_df = pd.DataFrame(query_data)
    print(prox.neighbors(query_df=query_df))

    # Test with Features list
    prox = Proximity(df, id_column="ID", features=["Feature1"], n_neighbors=2)
    print(prox.all_neighbors())

    # Create a sample DataFrame
    data = {
        "foo_id": ["a", "b", "c", "d", "e"],  # Testing string IDs
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)

    # Test with String Ids
    prox = Proximity(df, id_column="foo_id", features=["Feature1", "Feature2"], target="target", n_neighbors=3)
    print(prox.all_neighbors())

    # Test the neighbors method
    print(prox.neighbors(query_df=df.iloc[0:2], add_columns=["Feature1", "Feature2"]))

    # Time neighbors with all IDs versus calling all_neighbors
    import time

    start_time = time.time()
    df = prox.neighbors(query_df=df, include_self=False)
    end_time = time.time()
    print(f"Time taken for neighbors: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    df_all = prox.all_neighbors()
    end_time = time.time()
    print(f"Time taken for all_neighbors: {end_time - start_time:.4f} seconds")

    # Now compare the two dataframes
    print("Neighbors DataFrame:")
    print(df)
    print("\nAll Neighbors DataFrame:")
    print(df_all)
    # Check for any discrepancies
    if df.equals(df_all):
        print("The two DataFrames are equal :)")
    else:
        print("ERRPR: The two DataFrames are not equal!")