import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List
from workbench.algorithms.dataframe.proximity import Proximity


class FeaturesProximity(Proximity):
    def __init__(self, df: pd.DataFrame, id_column: str, features: List[str], n_neighbors: int = 10) -> None:
        """
        Initialize the FeaturesProximity class.

        Args:
            df (pd.DataFrame): DataFrame containing feature data.
            id_column (str): Name of the column used as an identifier.
            features (List[str]): List of feature column names to be used for neighbor computations.
            n_neighbors (int): Number of neighbors to compute.
        """
        if not features:
            raise ValueError("The 'features' list must be defined and contain at least one feature.")
        super().__init__(df, id_column=id_column, features=features, n_neighbors=n_neighbors)

    def _prepare_data(self) -> None:
        """
        Prepare the feature matrix by scaling numeric features.
        """
        # Scale features for better distance computation
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.df[self.features].values)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)


# __main__ Tests
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create a test DataFrame
    data = {
        "id": [1, 2, 3, 4],
        "Feature1": [0.1, 0.2, 0.3, 0.4],
        "Feature2": [0.5, 0.4, 0.3, 0.2],
        "Feature3": [1, 2, 3, 4],
    }
    df = pd.DataFrame(data)

    # Initialize the FeaturesProximity class
    features = ["Feature1", "Feature2", "Feature3"]
    proximity = FeaturesProximity(df, id_column="id", features=features, n_neighbors=2)

    # Test 1: All neighbors
    print("\n--- Test 1: All Neighbors ---")
    all_neighbors_df = proximity.all_neighbors(include_self=False)
    print(all_neighbors_df)

    # Test 2: Neighbors for a specific query
    print("\n--- Test 2: Neighbors for Query ID 1 ---")
    query_neighbors_df = proximity.neighbors(query_id=1)
    print(query_neighbors_df)

    # Test 3: Neighbors for a specific query with radius
    print("\n--- Test 3: Neighbors for Query ID 1 with Radius 2.0 ---")
    query_neighbors_radius_df = proximity.neighbors(query_id=1, radius=2.0)
    print(query_neighbors_radius_df)

    # Test 4: Edge case - Empty features list
    try:
        print("\n--- Test 4: Empty Features List ---")
        FeaturesProximity(df, id_column="id", features=[], n_neighbors=2)
    except ValueError as e:
        print(f"Expected error: {e}")

    # Test 5: Invalid feature name
    try:
        print("\n--- Test 5: Invalid Feature Name ---")
        FeaturesProximity(df, id_column="id", features=["NonexistentFeature"], n_neighbors=2)
    except KeyError as e:
        print(f"Expected error: {e}")
