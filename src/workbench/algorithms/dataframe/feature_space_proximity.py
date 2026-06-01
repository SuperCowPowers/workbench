import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from workbench.algorithms.dataframe.projection_2d import Projection2D

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity

# Set up logging
log = logging.getLogger("workbench")


class FeatureSpaceProximity(Proximity):
    """Proximity computations for numeric feature spaces using Euclidean distance.

    Implements the Proximity ABC contract:
        - `neighbors(id_or_ids)`     id-based lookups
        - `neighbors_from_query_df`  novel-input lookups (query_df must contain the
                                     same feature columns this model was built with)

    The `distance` column in results is standardized Euclidean distance (raw sklearn
    NearestNeighbors output). For visualization, call `project_2d()` explicitly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        features: List[str],
        target: Optional[str] = None,
        include_all_columns: bool = False,
    ):
        """
        Initialize the FeatureSpaceProximity class.

        Args:
            df: DataFrame containing data for neighbor computations.
            id_column: Name of the column used as the identifier.
            features: List of feature column names to be used for neighbor computations.
            target: Name of the target column. Defaults to None.
            include_all_columns: Include all DataFrame columns in neighbor results. Defaults to False.
        """
        self._raw_features = features
        super().__init__(
            df, id_column=id_column, features=features, target=target, include_all_columns=include_all_columns
        )

    def _prepare_data(self) -> None:
        """Filter unsupported features, encode categoricals, and drop NaN rows."""
        self.features = self._validate_features(self.df, self._raw_features)
        self._numeric_features = [
            feature for feature in self.features if pd.api.types.is_numeric_dtype(self.df[feature])
        ]
        self._categorical_features = [
            feature for feature in self.features if self._is_categorical_feature(self.df[feature])
        ]
        self.df = self.df.dropna(subset=self.features).copy()
        self._encoded_reference = self._encode_feature_frame(self.df, fit=True)

    def _validate_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove unsupported features and log warnings."""
        supported = []
        unsupported = []
        missing = [feature for feature in features if feature not in df.columns]

        for feature in features:
            if feature not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[feature]) or self._is_categorical_feature(df[feature]):
                supported.append(feature)
            else:
                unsupported.append(feature)

        if missing:
            log.warning(f"Features {missing} are missing from the DataFrame, excluding them")
        if unsupported:
            log.warning(f"Features {unsupported} have unsupported types, excluding them")
        if not supported:
            raise ValueError("No supported features remain for FeatureSpaceProximity")
        return supported

    @staticmethod
    def _is_categorical_feature(series: pd.Series) -> bool:
        """Return True for feature types that should be one-hot encoded."""
        return (
            str(series.dtype) == "category"
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        )

    def _encode_feature_frame(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Convert the original feature columns into the numeric matrix used by the NN model."""
        missing = [feature for feature in self.features if feature not in df.columns]
        if missing:
            raise ValueError(f"Query DataFrame is missing proximity features: {missing}")

        feature_df = df[self.features].copy()
        if self._categorical_features:
            feature_df = pd.get_dummies(feature_df, columns=self._categorical_features, dtype=float)

        if fit:
            self._encoded_features = feature_df.columns.tolist()
            return feature_df
        return feature_df.reindex(columns=self._encoded_features, fill_value=0.0)

    def _build_model(self) -> None:
        """Standardize features and fit Nearest Neighbors model."""
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self._encoded_reference)
        self.nn = NearestNeighbors().fit(X)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using the fitted scaler.

        For novel-input queries via `neighbors_from_query_df`, the query DataFrame
        must contain the same feature columns this model was built with.
        """
        return self.scaler.transform(self._encode_feature_frame(df))

    def project_2d(self) -> pd.DataFrame:
        """Project the numeric features to 2D for visualization (UMAP).

        Returns the reference DataFrame with 'x' / 'y' columns added.
        """
        if len(self._encoded_features) >= 2:
            projection_df = self._encoded_reference.copy()
            projection_df[self.id_column] = self.df[self.id_column].values
            projection_df = Projection2D().fit_transform(projection_df, features=self._encoded_features)
            self.df["x"] = projection_df["x"].values
            self.df["y"] = projection_df["y"].values
        return self.df


# Testing the FeatureSpaceProximity class
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Basic id-based lookup
    data = {
        "ID": [1, 2, 3, 4, 5],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "Feature3": [2.5, 2.4, 2.3, 2.3, np.nan],
    }
    df = pd.DataFrame(data)
    features = ["Feature1", "Feature2", "Feature3"]
    prox = FeatureSpaceProximity(df, id_column="ID", features=features)
    print("\nNeighbors for ID=1 (k=2):")
    print(prox.neighbors(1, n_neighbors=2))

    # Radius query
    print("\nNeighbors for ID=1 (radius=2.0):")
    print(prox.neighbors(1, radius=2.0))

    # Novel-input query
    novel = pd.DataFrame({"Feature1": [0.25], "Feature2": [0.35], "Feature3": [2.35]})
    print("\nNovel-input query:")
    print(prox.neighbors_from_query_df(novel, n_neighbors=3))

    # String IDs + target + include_all_columns
    data = {
        "id": ["a", "b", "c", "d", "e"],
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)
    prox = FeatureSpaceProximity(
        df, id_column="id", features=["Feature1", "Feature2"], target="target", include_all_columns=True
    )
    print("\nString IDs (batch):")
    print(prox.neighbors(["a", "b"]))
