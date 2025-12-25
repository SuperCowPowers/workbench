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
    """Proximity computations for numeric feature spaces using Euclidean distance."""

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
        # Validate and filter features before calling parent init
        self._raw_features = features
        super().__init__(
            df, id_column=id_column, features=features, target=target, include_all_columns=include_all_columns
        )

    def _prepare_data(self) -> None:
        """Filter out non-numeric features and drop NaN rows."""
        # Validate features
        self.features = self._validate_features(self.df, self._raw_features)

        # Drop NaN rows for the features we're using
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
        """Transform features using the fitted scaler."""
        return self.scaler.transform(df[self.features])

    def _project_2d(self) -> None:
        """Project the numeric features to 2D for visualization."""
        if len(self.features) >= 2:
            self.df = Projection2D().fit_transform(self.df, features=self.features)


# Testing the FeatureSpaceProximity class
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

    # Test the FeatureSpaceProximity class
    features = ["Feature1", "Feature2", "Feature3"]
    prox = FeatureSpaceProximity(df, id_column="ID", features=features)
    print(prox.neighbors(1, n_neighbors=2))

    # Test the neighbors method with radius
    print(prox.neighbors(1, radius=2.0))

    # Test with Features list
    prox = FeatureSpaceProximity(df, id_column="ID", features=["Feature1"])
    print(prox.neighbors(1))

    # Create a sample DataFrame
    data = {
        "id": ["a", "b", "c", "d", "e"],  # Testing string IDs
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)

    # Test with String Ids
    prox = FeatureSpaceProximity(
        df,
        id_column="id",
        features=["Feature1", "Feature2"],
        target="target",
        include_all_columns=True,
    )
    print(prox.neighbors(["a", "b"]))

    # Test duplicate IDs
    data = {
        "id": ["a", "b", "c", "d", "d"],  # Duplicate ID (d)
        "Feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)
    prox = FeatureSpaceProximity(df, id_column="id", features=["Feature1", "Feature2"], target="target")
    print(df.equals(prox.df))

    # Test on real data from Workbench
    from workbench.api import FeatureSet, Model

    fs = FeatureSet("aqsol_features")
    model = Model("aqsol-regression")
    features = model.features()
    df = fs.pull_dataframe()
    prox = FeatureSpaceProximity(df, id_column=fs.id_column, features=model.features(), target=model.target())
    print("\n" + "=" * 80)
    print("Testing Neighbors...")
    print("=" * 80)
    test_id = df[fs.id_column].tolist()[0]
    print(f"\nNeighbors for ID {test_id}:")
    print(prox.neighbors(test_id))

    print("\n" + "=" * 80)
    print("Testing isolated_compounds...")
    print("=" * 80)

    # Test isolated data in the top 1%
    isolated_1pct = prox.isolated(top_percent=1.0)
    print(f"\nTop 1% most isolated compounds (n={len(isolated_1pct)}):")
    print(isolated_1pct)

    # Test isolated data in the top 5%
    isolated_5pct = prox.isolated(top_percent=5.0)
    print(f"\nTop 5% most isolated compounds (n={len(isolated_5pct)}):")
    print(isolated_5pct)

    print("\n" + "=" * 80)
    print("Testing target_gradients...")
    print("=" * 80)

    # Test with different parameters
    gradients_1pct = prox.target_gradients(top_percent=1.0, min_delta=1.0)
    print(f"\nTop 1% target gradients (min_delta=5.0) (n={len(gradients_1pct)}):")
    print(gradients_1pct)

    gradients_5pct = prox.target_gradients(top_percent=5.0, min_delta=5.0)
    print(f"\nTop 5% target gradients (min_delta=5.0) (n={len(gradients_5pct)}):")
    print(gradients_5pct)

    # Test proximity_stats
    print("\n" + "=" * 80)
    print("Testing proximity_stats...")
    print("=" * 80)
    stats = prox.proximity_stats()
    print(stats)

    # Plot the distance distribution using pandas
    print("\n" + "=" * 80)
    print("Plotting distance distribution...")
    print("=" * 80)
    prox.df["nn_distance"].hist(bins=50, figsize=(10, 6), edgecolor="black")

    # Visualize the 2D projection
    print("\n" + "=" * 80)
    print("Visualizing 2D Projection...")
    print("=" * 80)
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    unit_test = PluginUnitTest(ScatterPlot, input_data=prox.df[:1000], x="x", y="y", color=model.target())
    unit_test.run()
