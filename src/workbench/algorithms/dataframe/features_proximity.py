import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List
from workbench.algorithms.dataframe.proximity import Proximity


class FeaturesProximity(Proximity):
    def __init__(
        self, df: pd.DataFrame, id_column: str, features: List[str], target: str = None, n_neighbors: int = 10
    ) -> None:
        """
        Initialize the FeaturesProximity class.

        Args:
            df (pd.DataFrame): DataFrame containing feature data.
            id_column (str): Name of the column used as an identifier.
            features (List[str]): List of feature column names to be used for neighbor computations.
            target (str): Optional name of the target column.
            n_neighbors (int): Number of neighbors to compute.
        """
        self.target = target
        if not features:
            raise ValueError("The 'features' list must be defined and contain at least one feature.")
        super().__init__(df, id_column=id_column, features=features, n_neighbors=n_neighbors)

    def _prepare_data(self) -> None:
        """
        Prepare the feature matrix by scaling numeric features.
        """
        # Scale features for better distance computation
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.data[self.features].values)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)

    @classmethod
    def from_model(cls, model) -> "FeaturesProximity":
        """Create a FeaturesProximity instance from a Workbench model object.

        Args:
            model (Model): A Workbench model object.

        Returns:
            FeaturesProximity: A new instance of the FeaturesProximity class.
        """
        from workbench.api import FeatureSet

        # Extract necessary attributes from the Workbench model
        fs = FeatureSet(model.get_input())
        features = model.features()
        target = model.target()

        # Retrieve the training DataFrame from the feature set
        df = fs.view("training").pull_dataframe()

        # Create and return a new instance of FeatureSpaceProximityDeprecated
        return cls(df=df, id_column=fs.id_column, features=features, target=target)


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

    # Test 4: From a Workbench model
    from workbench.api import Model

    m = Model("abalone-regression")
    proximity = FeaturesProximity.from_model(m)

    # Neighbors Test using a single query ID
    single_query_id = 5
    single_query_neighbors = proximity.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)
