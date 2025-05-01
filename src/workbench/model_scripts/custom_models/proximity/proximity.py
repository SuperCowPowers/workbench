import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Union, List, Dict
import logging
import pickle
import os
import json
from pathlib import Path
from enum import Enum

# Set up logging
log = logging.getLogger("workbench")


# ^Enumerated^ Proximity Types (distance or similarity)
class ProximityType(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"


class Proximity:
    def __init__(
        self,
        df: pd.DataFrame,
        id_column: Union[int, str],
        features: List[str],
        target: str = None,
        track_columns: List[str] = None,
        n_neighbors: int = 10,
    ):
        """
        Initialize the Proximity class.

        Args:
            df (pd.DataFrame): DataFrame containing data for neighbor computations.
            id_column (Union[int, str]): Name of the column used as an identifier.
            features (List[str]): List of feature column names to be used for neighbor computations.
            target (str, optional): Name of the target column. Defaults to None.
            track_columns (List[str], optional): Additional columns to track in results. Defaults to None.
            n_neighbors (int): Number of neighbors to compute. Defaults to 10.
        """
        self.df = df.dropna(subset=features).copy()
        self.id_column = id_column
        self.n_neighbors = min(n_neighbors, len(self.df) - 1)
        self.target = target
        self.features = features
        self.scaler = None
        self.X = None
        self.nn = None
        self.proximity_type = None
        self.track_columns = track_columns or []

        # Right now we only support numeric features, so remove any columns that are not numeric
        non_numeric_features = self.df[self.features].select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric_features:
            log.warning(f"Non-numeric features {non_numeric_features} aren't currently supported...")
            self.features = [f for f in self.features if f not in non_numeric_features]

        # Build the proximity model
        self.build_proximity_model()

    def build_proximity_model(self) -> None:
        """Standardize features and fit Nearest Neighbors model.
        Note: This method can be overridden in subclasses for custom behavior."""
        self.proximity_type = ProximityType.DISTANCE
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[self.features])
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
            query_id = self.df.iloc[i][self.id_column]

            # Process neighbors
            for neighbor_idx, dist in zip(nbrs, dists):
                # Skip self (neighbor index == current row index)
                if neighbor_idx == i:
                    continue
                results.append(self._build_neighbor_result(query_id=query_id, neighbor_idx=neighbor_idx, distance=dist))

        return pd.DataFrame(results)

    def neighbors(
        self,
        query_df: pd.DataFrame,
        radius: float = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """
        Return neighbors for rows in a query DataFrame.

        Args:
            query_df: DataFrame containing query points
            radius: If provided, find all neighbors within this radius
            include_self: Whether to include self in results (if present)

        Returns:
            DataFrame containing neighbors and distances

        Note: The query DataFrame must include the feature columns. The id_column is optional.
        """
        # Check if all required features are present
        missing = set(self.features) - set(query_df.columns)
        if missing:
            raise ValueError(f"Query DataFrame is missing required feature columns: {missing}")

        # Check if id_column is present
        id_column_present = self.id_column in query_df.columns

        # None of the features can be NaNs, so report rows with NaNs and then drop them
        rows_with_nan = query_df[self.features].isna().any(axis=1)

        # Print the ID column for rows with NaNs
        if rows_with_nan.any():
            log.warning(f"Found {rows_with_nan.sum()} rows with NaNs in feature columns:")
            log.warning(query_df.loc[rows_with_nan, self.id_column])

        # Drop rows with NaNs in feature columns and reassign to query_df
        query_df = query_df.dropna(subset=self.features)

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
            # Use the ID from the query DataFrame if available, otherwise use the row index
            query_id = query_df.iloc[i][self.id_column] if id_column_present else f"query_{i}"

            for neighbor_idx, dist in zip(nbrs, dists):
                # Skip if the neighbor is the query itself and include_self is False
                neighbor_id = self.df.iloc[neighbor_idx][self.id_column]
                if not include_self and neighbor_id == query_id:
                    continue

                all_results.append(
                    self._build_neighbor_result(query_id=query_id, neighbor_idx=neighbor_idx, distance=dist)
                )

        return pd.DataFrame(all_results)

    def _build_neighbor_result(self, query_id, neighbor_idx: int, distance: float) -> Dict:
        """
        Internal: Build a result dictionary for a single neighbor.

        Args:
            query_id: ID of the query point
            neighbor_idx: Index of the neighbor in the original DataFrame
            distance: Distance between query and neighbor

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
        relevant_cols += self.track_columns

        # Add values for each relevant column that exists in the dataframe
        for col in filter(lambda c: c in self.df.columns, relevant_cols):
            neighbor_info[col] = self.df.iloc[neighbor_idx][col]

        return neighbor_info

    def serialize(self, directory: str) -> None:
        """
        Serialize the Proximity model to a directory.

        Args:
            directory: Directory path to save the model components
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save metadata
        metadata = {
            "id_column": self.id_column,
            "features": self.features,
            "target": self.target,
            "track_columns": self.track_columns,
            "n_neighbors": self.n_neighbors,
        }

        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Save the DataFrame
        self.df.to_pickle(os.path.join(directory, "df.pkl"))

        # Save the scaler and nearest neighbors model
        with open(os.path.join(directory, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(directory, "nn_model.pkl"), "wb") as f:
            pickle.dump(self.nn, f)

        log.info(f"Proximity model serialized to {directory}")

    @classmethod
    def deserialize(cls, directory: str) -> "Proximity":
        """
        Deserialize a Proximity model from a directory.

        Args:
            directory: Directory path containing the serialized model components

        Returns:
            Proximity: A new Proximity instance
        """
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist or is not a directory")

        # Load metadata
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Load DataFrame
        df_path = os.path.join(directory, "df.pkl")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"DataFrame file not found at {df_path}")
        df = pd.read_pickle(df_path)

        # Create instance but skip _prepare_data
        instance = cls.__new__(cls)
        instance.df = df
        instance.id_column = metadata["id_column"]
        instance.features = metadata["features"]
        instance.target = metadata["target"]
        instance.track_columns = metadata["track_columns"]
        instance.n_neighbors = metadata["n_neighbors"]

        # Load scaler and nn model
        with open(os.path.join(directory, "scaler.pkl"), "rb") as f:
            instance.scaler = pickle.load(f)

        with open(os.path.join(directory, "nn_model.pkl"), "rb") as f:
            instance.nn = pickle.load(f)

        # Load X from scaler transform
        instance.X = instance.scaler.transform(instance.df[instance.features])

        log.info(f"Proximity model deserialized from {directory}")
        return instance


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
    prox = Proximity(
        df,
        id_column="foo_id",
        features=["Feature1", "Feature2"],
        target="target",
        track_columns=["Feature1", "Feature2"],
        n_neighbors=3,
    )
    print(prox.all_neighbors())

    # Test the neighbors method
    print(prox.neighbors(query_df=df.iloc[0:2]))

    # Time neighbors with all IDs versus calling all_neighbors
    import time

    start_time = time.time()
    prox_df = prox.neighbors(query_df=df, include_self=False)
    end_time = time.time()
    print(f"Time taken for neighbors: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    prox_df_all = prox.all_neighbors()
    end_time = time.time()
    print(f"Time taken for all_neighbors: {end_time - start_time:.4f} seconds")

    # Now compare the two dataframes
    print("Neighbors DataFrame:")
    print(prox_df)
    print("\nAll Neighbors DataFrame:")
    print(prox_df_all)
    # Check for any discrepancies
    if prox_df.equals(prox_df_all):
        print("The two DataFrames are equal :)")
    else:
        print("ERROR: The two DataFrames are not equal!")

    # Test querying without the id_column
    df_no_id = df.drop(columns=["foo_id"])
    print(prox.neighbors(query_df=df_no_id, include_self=False))

    # Test duplicate IDs
    data = {
        "foo_id": ["a", "b", "c", "d", "d"],  # Duplicate ID (d)
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)
    prox = Proximity(df, id_column="foo_id", features=["Feature1", "Feature2"], target="target", n_neighbors=3)
    print(df.equals(prox.df))

    # Test with a categorical feature
    from workbench.api import FeatureSet, Model

    fs = FeatureSet("abalone_features")
    model = Model("abalone-regression")
    df = fs.pull_dataframe()
    prox = Proximity(df, id_column=fs.id_column, features=model.features(), target=model.target())
    print(prox.neighbors(query_df=df[0:2]))
