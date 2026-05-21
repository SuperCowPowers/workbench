import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional
import logging

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity
from workbench.algorithms.dataframe.projection_2d import Projection2D

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
        """Filter out non-numeric features and drop NaN rows."""
        self.features = self._validate_features(self.df, self._raw_features)
        self.df = self.df.dropna(subset=self.features).copy()

    def _validate_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove non-numeric features and log warnings."""
        non_numeric = [f for f in features if f not in df.select_dtypes(include=["number"]).columns]
        if non_numeric:
            log.warning(f"Non-numeric features {non_numeric} aren't currently supported, excluding them")
        return [f for f in features if f not in non_numeric]

    def _build_model(self) -> None:
        """Standardize features and fit Nearest Neighbors model."""
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self.df[self.features])
        self.nn = NearestNeighbors().fit(X)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using the fitted scaler.

        For novel-input queries via `neighbors_from_query_df`, the query DataFrame
        must contain the same feature columns this model was built with.
        """
        return self.scaler.transform(df[self.features])

    def project_2d(self) -> pd.DataFrame:
        """Project the numeric features to 2D for visualization (UMAP).

        Returns the reference DataFrame with 'x' / 'y' columns added.
        """
        if len(self.features) >= 2:
            self.df = Projection2D().fit_transform(self.df, features=self.features)
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
