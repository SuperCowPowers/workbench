from typing import Union
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging


class FeatureSpaceProximity:
    def __init__(self, df: pd.DataFrame, features: list, id_column: str, neighbors: int = 10):
        """FeatureSpaceProximity: A class for managing feature space proximity using KNN.

        Args:
            df: Pandas DataFrame
            features: List of feature column names
            id_column: Name of the ID column
            neighbors: Number of neighbors to use in the KNN model (default: 10)
        """
        self.log = logging.getLogger("sageworks")
        self.df = df.copy()
        self.features = features
        self.id_column = id_column

        # Standardize the feature values and build the KNN model
        self.scaler = StandardScaler().fit(df[features])
        scaled_features = self.scaler.transform(df[features])
        self.knn_model = NearestNeighbors(n_neighbors=neighbors, algorithm="auto").fit(scaled_features)

    def get_neighbor_indices_and_distances(self):
        """Retrieve neighbor indices and distances for all points in the dataset."""
        distances, indices = self.knn_model.kneighbors()
        return indices, distances

    def neighbors(self, query_id: Union[str, int], include_self: bool = True) -> pd.DataFrame:
        """Return neighbors of the given query ID.

        Args:
            query_id (Union[str, int]): The ID of the query point.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            pd.DataFrame: Filtered DataFrame that includes the query ID and its neighbors.
        """
        if query_id not in self.df[self.id_column].values:
            self.log.warning(f"Query ID '{query_id}' not found in the DataFrame. Returning an empty DataFrame.")
            return pd.DataFrame()

        query_df = self.df[self.df[self.id_column] == query_id]
        neighbors_info_df = self.neighbors_bulk(query_df, include_self)
        neighbor_ids = neighbors_info_df["neighbor_ids"].iloc[0]
        neighbor_distances = neighbors_info_df["neighbor_distances"].iloc[0]

        neighbors_df = self.df[self.df[self.id_column].isin(neighbor_ids)]
        neighbors_df = neighbors_df.set_index(self.id_column).reindex(neighbor_ids).reset_index()
        neighbors_df["knn_distance"] = neighbor_distances

        return neighbors_df

    def neighbors_bulk(self, query_df: pd.DataFrame, include_self: bool = False) -> pd.DataFrame:
        """Return neighbors for each row in the given query dataframe."""
        query_scaled = self.scaler.transform(query_df[self.features])
        distances, indices = self.knn_model.kneighbors(query_scaled)
        query_ids = query_df[self.id_column].values
        neighbor_ids = [[self.df.iloc[idx][self.id_column] for idx in index_list] for index_list in indices]
        neighbor_distances = [list(dist_list) for dist_list in distances]

        for i, query_id in enumerate(query_ids):
            if query_id in neighbor_ids[i] and not include_self:
                idx_to_remove = neighbor_ids[i].index(query_id)
                neighbor_ids[i].pop(idx_to_remove)
                neighbor_distances[i].pop(idx_to_remove)

        result_df = pd.DataFrame({"query_id": query_ids, "neighbor_ids": neighbor_ids, "neighbor_distances": neighbor_distances})
        return result_df

    def query_by_radius(self, query_df: pd.DataFrame, radius: float) -> pd.DataFrame:
        """Return neighbors within a specified radius."""
        query_scaled = self.scaler.transform(query_df[self.features])
        radius_indices = self.knn_model.radius_neighbors(query_scaled, radius=radius, return_distance=False)

        query_ids = query_df[self.id_column].values
        neighbor_ids = [[self.df.iloc[idx][self.id_column] for idx in index_list] for index_list in radius_indices]

        result_df = pd.DataFrame({"query_id": query_ids, "neighbor_ids": neighbor_ids})
        return result_df


# Testing the FeatureSpaceProximity class with separate training and test/evaluation dataframes
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Training Data (20 rows)
    training_data = {
        "ID": [f"id_{i}" for i in range(20)],
        "feat1": [1.0, 1.1, 1.2, 3.0, 4.0, 1.0, 1.2, 3.1, 4.2, 4.3, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.3, 3.2, 4.3, 4.4, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3, 0.5, 0.4, 0.5, 0.6],
        "feat3": [0.1, 0.2, 0.3, 1.6, 2.5, 0.2, 0.3, 1.7, 2.6, 2.7, 1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.5],
        "target": [10, 11, 12, 20, 30, 10, 12, 21, 31, 32, 15, 16, 17, 5, 6, 7, 8, 9, 10, 11],
    }
    training_df = pd.DataFrame(training_data)

    # Create a classification column "class" by cutting the target into "low", "medium", "high"
    training_df["class"] = pd.cut(training_df["target"], bins=3, labels=["low", "medium", "high"])

    # Test Data (5 rows)
    test_data = {
        "ID": [f"test_id_{i}" for i in range(5)],
        "feat1": [0.8, 1.5, 3.5, 2.3, 4.5],
        "feat2": [0.8, 1.4, 3.8, 2.4, 4.6],
        "feat3": [0.4, 0.5, 2.0, 1.2, 2.8],
        "target": [9, 13, 25, 18, 35],  # Target values for regression testing
    }
    test_df = pd.DataFrame(test_data)

    # Create a classification column for the test data
    test_df["class"] = pd.cut(test_df["target"], bins=3, labels=["low", "medium", "high"])

    # Regression Test using Training and Test DataFrames
    reg_spider = FeatureSpaceProximity(
        training_df, ["feat1", "feat2", "feat3"], target="target", id_column="ID", classification=False
    )
    reg_predictions = reg_spider.predict(test_df)
    print("Regression Predictions (Test Data):\n", reg_predictions)

    # Regression Neighbors Test
    reg_neighbors = reg_spider.neighbors_bulk(test_df)
    print("\nRegression Neighbors (Test Data):\n", reg_neighbors)

    # Classification Test using Training and Test DataFrames
    class_spider = FeatureSpaceProximity(
        training_df,
        ["feat1", "feat2", "feat3"],
        id_column="ID",
        target="class",
        classification=True,
        class_labels=["low", "medium", "high"],
    )
    class_predictions = class_spider.predict(test_df)
    class_probs = class_spider.predict_proba(test_df)
    print("\nClassification Predictions (Test Data):\n", class_predictions)
    print("Classification Probabilities (Test Data, Ordered):\n", class_probs)

    # Classification Neighbors Test
    class_neighbors = class_spider.neighbors_bulk(test_df)
    print("\nClassification Neighbors (Test Data):\n", class_neighbors)

    # Neighbor Test using a single query ID
    single_query_id = "id_5"
    single_query_neighbors = class_spider.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)

    # Neighbor Indices and Distances Test
    indices, distances = class_spider.get_neighbor_indices_and_distances()
    print("\nNeighbor Indices (Training Data):\n", indices)
    print("\nNeighbor Distances (Training Data):\n", distances)
