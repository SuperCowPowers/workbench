import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional
import logging
import pickle
import json
from pathlib import Path
from enum import Enum

# Set up logging
log = logging.getLogger("workbench")


class ProximityType(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"


class Proximity:
    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        features: List[str],
        target: Optional[str] = None,
        track_columns: Optional[List[str]] = None,
        n_neighbors: int = 10,
    ):
        """
        Initialize the Proximity class.

        Args:
            df: DataFrame containing data for neighbor computations.
            id_column: Name of the column used as the identifier.
            features: List of feature column names to be used for neighbor computations.
            target: Name of the target column. Defaults to None.
            track_columns: Additional columns to track in results. Defaults to None.
            n_neighbors: Number of neighbors to compute. Defaults to 10.
        """
        self.id_column = id_column
        self.target = target
        self.track_columns = track_columns or []
        self.proximity_type = None
        self.scaler = None
        self.X = None
        self.nn = None

        # Filter out non-numeric features
        self.features = self._validate_features(df, features)

        # Drop NaN rows and set up DataFrame
        self.df = df.dropna(subset=self.features).copy()
        self.n_neighbors = min(n_neighbors, len(self.df) - 1)

        # Build the proximity model
        self.build_proximity_model()

    def _validate_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove non-numeric features and log warnings."""
        non_numeric = df[features].select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            log.warning(f"Non-numeric features {non_numeric} aren't currently supported...")
            return [f for f in features if f not in non_numeric]
        return features

    def build_proximity_model(self) -> None:
        """Standardize features and fit Nearest Neighbors model."""
        self.proximity_type = ProximityType.DISTANCE
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[self.features])
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    def all_neighbors(self) -> pd.DataFrame:
        """
        Compute nearest neighbors for all rows in the dataset.

        Returns:
            DataFrame of neighbors and their distances.
        """
        distances, indices = self.nn.kneighbors(self.X)

        results = [
            self._build_neighbor_result(
                query_id=self.df.iloc[i][self.id_column], neighbor_idx=neighbor_idx, distance=dist
            )
            for i, (dists, nbrs) in enumerate(zip(distances, indices))
            for neighbor_idx, dist in zip(nbrs, dists)
            if neighbor_idx != i  # Skip self
        ]

        return pd.DataFrame(results)

    def neighbors(
        self,
        id_or_ids,
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """
        Return neighbors for ID(s) from the existing dataset.

        Args:
            id_or_ids: Single ID or list of IDs to look up
            n_neighbors: Number of neighbors to return (default: 5)
            radius: If provided, find all neighbors within this radius
            include_self: Whether to include self in results (if present)

        Returns:
            DataFrame containing neighbors and distances
        """
        # Normalize to list
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else id_or_ids

        # Validate IDs exist
        missing_ids = set(ids) - set(self.df[self.id_column])
        if missing_ids:
            raise ValueError(f"IDs not found in dataset: {missing_ids}")

        # Filter to requested IDs and preserve order
        query_df = self.df[self.df[self.id_column].isin(ids)]
        query_df = query_df.set_index(self.id_column).loc[ids].reset_index()

        # Use the core implementation
        return self.find_neighbors(query_df, n_neighbors=n_neighbors, radius=radius, include_self=include_self)

    def find_neighbors(
        self,
        query_df: pd.DataFrame,
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """
        Return neighbors for rows in a query DataFrame.

        Args:
            query_df: DataFrame containing query points
            n_neighbors: Number of neighbors to return (default: 5)
            radius: If provided, find all neighbors within this radius
            include_self: Whether to include self in results (if present)

        Returns:
            DataFrame containing neighbors and distances
        """
        # Validate features
        missing = set(self.features) - set(query_df.columns)
        if missing:
            raise ValueError(f"Query DataFrame is missing required feature columns: {missing}")

        id_column_present = self.id_column in query_df.columns

        # Handle NaN rows
        query_df = self._handle_nan_rows(query_df, id_column_present)

        # Transform query features
        X_query = self.scaler.transform(query_df[self.features])

        # Get neighbors
        if radius is not None:
            distances, indices = self.nn.radius_neighbors(X_query, radius=radius)
        else:
            distances, indices = self.nn.kneighbors(X_query, n_neighbors=n_neighbors)

        # Build results
        results = []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            query_id = query_df.iloc[i][self.id_column] if id_column_present else f"query_{i}"

            for neighbor_idx, dist in zip(nbrs, dists):
                neighbor_id = self.df.iloc[neighbor_idx][self.id_column]

                # Skip if neighbor is self and include_self is False
                if not include_self and neighbor_id == query_id:
                    continue

                results.append(self._build_neighbor_result(query_id=query_id, neighbor_idx=neighbor_idx, distance=dist))

        results_df = pd.DataFrame(results).sort_values([self.id_column, "distance"]).reset_index(drop=True)
        return results_df

    def _handle_nan_rows(self, query_df: pd.DataFrame, id_column_present: bool) -> pd.DataFrame:
        """Drop rows with NaN values in feature columns and log warnings."""
        rows_with_nan = query_df[self.features].isna().any(axis=1)

        if rows_with_nan.any():
            log.warning(f"Found {rows_with_nan.sum()} rows with NaNs in feature columns:")
            if id_column_present:
                log.warning(query_df.loc[rows_with_nan, self.id_column])

        return query_df.dropna(subset=self.features)

    def _build_neighbor_result(self, query_id, neighbor_idx: int, distance: float) -> Dict:
        """
        Build a result dictionary for a single neighbor.

        Args:
            query_id: ID of the query point
            neighbor_idx: Index of the neighbor in the original DataFrame
            distance: Distance between query and neighbor

        Returns:
            Dictionary containing neighbor information
        """
        neighbor_id = self.df.iloc[neighbor_idx][self.id_column]
        neighbor_row = self.df.iloc[neighbor_idx]

        # Start with basic info
        result = {
            self.id_column: query_id,
            "neighbor_id": neighbor_id,
            "distance": distance,
        }

        # Columns to automatically include if they exist
        auto_include = (
            ([self.target, "prediction"] if self.target else [])
            + self.track_columns
            + [col for col in self.df.columns if "_proba" in col or "residual" in col or col == "outlier"]
        )

        # Add values for existing columns
        for col in auto_include:
            if col in self.df.columns:
                result[col] = neighbor_row[col]

        # Truncate very small distances to zero
        result["distance"] = 0.0 if distance < 1e-7 else distance
        return result

    def serialize(self, directory: str) -> None:
        """
        Serialize the Proximity model to a directory.

        Args:
            directory: Directory path to save the model components
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "id_column": self.id_column,
            "features": self.features,
            "target": self.target,
            "track_columns": self.track_columns,
            "n_neighbors": self.n_neighbors,
        }

        (dir_path / "metadata.json").write_text(json.dumps(metadata))

        # Save DataFrame
        self.df.to_pickle(dir_path / "df.pkl")

        # Save models
        with open(dir_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(dir_path / "nn_model.pkl", "wb") as f:
            pickle.dump(self.nn, f)

        log.info(f"Proximity model serialized to {directory}")

    @classmethod
    def deserialize(cls, directory: str) -> "Proximity":
        """
        Deserialize a Proximity model from a directory.

        Args:
            directory: Directory path containing the serialized model components

        Returns:
            A new Proximity instance
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist or is not a directory")

        # Load metadata
        metadata = json.loads((dir_path / "metadata.json").read_text())

        # Load DataFrame
        df_path = dir_path / "df.pkl"
        if not df_path.exists():
            raise FileNotFoundError(f"DataFrame file not found at {df_path}")
        df = pd.read_pickle(df_path)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.df = df
        instance.id_column = metadata["id_column"]
        instance.features = metadata["features"]
        instance.target = metadata["target"]
        instance.track_columns = metadata["track_columns"]
        instance.n_neighbors = metadata["n_neighbors"]

        # Load models
        with open(dir_path / "scaler.pkl", "rb") as f:
            instance.scaler = pickle.load(f)

        with open(dir_path / "nn_model.pkl", "rb") as f:
            instance.nn = pickle.load(f)

        # Restore X
        instance.X = instance.scaler.transform(instance.df[instance.features])
        instance.proximity_type = ProximityType.DISTANCE

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
    print(prox.neighbors(1))

    # Test the neighbors method with radius
    print(prox.neighbors(1, radius=2.0))

    # Test with data that isn't in the 'train' dataframe
    query_data = {
        "ID": [6],
        "Feature1": [0.31],
        "Feature2": [0.31],
        "Feature3": [2.31],
    }
    query_df = pd.DataFrame(query_data)
    print(prox.find_neighbors(query_df=query_df))  # For new data we use find_neighbors()

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
    print(prox.neighbors(["a", "b"]))

    # Time neighbors with all IDs versus calling all_neighbors
    import time

    start_time = time.time()
    prox_df = prox.find_neighbors(query_df=df, include_self=False)
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
    print(prox.find_neighbors(query_df=df_no_id, include_self=False))

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
    features = model.features()
    df = fs.pull_dataframe()
    prox = Proximity(
        df, id_column=fs.id_column, features=model.features(), target=model.target(), track_columns=features
    )
    print(prox.find_neighbors(query_df=df[0:2]))
