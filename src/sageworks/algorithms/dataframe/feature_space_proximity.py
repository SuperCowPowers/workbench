"""FeatureSpaceProximity: A class for neighbor lookups using KNN with optional target information."""

from typing import Union
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging


class FeatureSpaceProximity:
    def __init__(self, df: pd.DataFrame, features: list, id_column: str, neighbors: int = 10, target: str = None):
        """FeatureSpaceProximity: A class for neighbor lookups using KNN with optional target information.

        Args:
            df: Pandas DataFrame
            features: List of feature column names
            id_column: Name of the ID column
            neighbors: Number of neighbors to use in the KNN model (default: 10)
            target: Optional name of the target column to include target-based functionality (default: None)
        """
        self.log = logging.getLogger("sageworks")
        self.df = df.copy()
        self.features = features
        self.id_column = id_column
        self.target = target

        # Standardize the feature values and build the KNN model
        self.scaler = StandardScaler().fit(df[features])
        scaled_features = self.scaler.transform(df[features])
        self.knn_model = NearestNeighbors(n_neighbors=neighbors, algorithm="auto").fit(scaled_features)

    def get_neighbor_indices_and_distances(self):
        """Retrieve neighbor indices and distances for all points in the dataset."""
        distances, indices = self.knn_model.kneighbors()
        return indices, distances

    def neighbors(self, query_id: Union[str, int], radius: float = None, include_self: bool = True) -> pd.DataFrame:
        """Return neighbors of the given query ID, either by fixed neighbors or within a radius.

        Args:
            query_id (Union[str, int]): The ID of the query point.
            radius (float): Optional radius within which neighbors are to be searched, else use fixed neighbors.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            pd.DataFrame: Filtered DataFrame that includes the query ID, its neighbors, and optionally target values.
        """
        if query_id not in self.df[self.id_column].values:
            self.log.warning(f"Query ID '{query_id}' not found in the DataFrame. Returning an empty DataFrame.")
            return pd.DataFrame()

        # Get a single-row DataFrame for the query ID
        query_df = self.df[self.df[self.id_column] == query_id]

        # Use the neighbors_bulk method with the appropriate radius
        neighbors_info_df = self.neighbors_bulk(query_df, radius=radius, include_self=include_self)

        # Extract the neighbor IDs and distances from the results
        neighbor_ids = neighbors_info_df["neighbor_ids"].iloc[0]
        neighbor_distances = neighbors_info_df["neighbor_distances"].iloc[0]

        # Sort neighbors by distance (ascending order)
        sorted_neighbors = sorted(zip(neighbor_ids, neighbor_distances), key=lambda x: x[1])
        sorted_ids, sorted_distances = zip(*sorted_neighbors)

        # Filter the internal DataFrame to include only the sorted neighbors
        neighbors_df = self.df[self.df[self.id_column].isin(sorted_ids)]
        neighbors_df = neighbors_df.set_index(self.id_column).reindex(sorted_ids).reset_index()
        neighbors_df["knn_distance"] = sorted_distances
        return neighbors_df

    def neighbors_bulk(self, query_df: pd.DataFrame, radius: float = None, include_self: bool = False) -> pd.DataFrame:
        """Return neighbors for each row in the given query dataframe, either by fixed neighbors or within a radius.

        Args:
            query_df: Pandas DataFrame with the same features as the training data.
            radius: Optional radius within which neighbors are to be searched, else use fixed neighbors.
            include_self: Boolean indicating whether to include the query ID in the neighbor results.

        Returns:
            pd.DataFrame: DataFrame with query ID, neighbor IDs, neighbor targets, and neighbor distances.
        """
        # Scale the query data using the same scaler as the training data
        query_scaled = self.scaler.transform(query_df[self.features])

        # Retrieve neighbors based on radius or standard neighbors
        if radius is not None:
            distances, indices = self.knn_model.radius_neighbors(query_scaled, radius=radius)
        else:
            distances, indices = self.knn_model.kneighbors(query_scaled)

        # Collect neighbor information (IDs, target values, and distances)
        query_ids = query_df[self.id_column].values
        neighbor_ids = [[self.df.iloc[idx][self.id_column] for idx in index_list] for index_list in indices]
        neighbor_targets = (
            [
                [self.df.loc[self.df[self.id_column] == neighbor, self.target].values[0] for neighbor in index_list]
                for index_list in neighbor_ids
            ]
            if self.target
            else None
        )
        neighbor_distances = [list(dist_list) for dist_list in distances]

        # Automatically remove the query ID itself from the neighbor results if include_self is False
        for i, query_id in enumerate(query_ids):
            if query_id in neighbor_ids[i] and not include_self:
                idx_to_remove = neighbor_ids[i].index(query_id)
                neighbor_ids[i].pop(idx_to_remove)
                neighbor_distances[i].pop(idx_to_remove)
                if neighbor_targets:
                    neighbor_targets[i].pop(idx_to_remove)

            # Sort neighbors by distance (ascending order)
            sorted_neighbors = sorted(zip(neighbor_ids[i], neighbor_distances[i]), key=lambda x: x[1])
            neighbor_ids[i], neighbor_distances[i] = list(zip(*sorted_neighbors)) if sorted_neighbors else ([], [])
            if neighbor_targets:
                neighbor_targets[i] = [
                    self.df.loc[self.df[self.id_column] == neighbor, self.target].values[0]
                    for neighbor in neighbor_ids[i]
                ]

        # Create and return a results DataFrame with the updated neighbor information
        result_df = pd.DataFrame(
            {
                "query_id": query_ids,
                "neighbor_ids": neighbor_ids,
                "neighbor_distances": neighbor_distances,
            }
        )

        if neighbor_targets:
            result_df["neighbor_targets"] = neighbor_targets

        return result_df

    def target_summary(self, query_id: Union[str, int]) -> pd.DataFrame:
        """Provide a summary of target values in the neighborhood of the given query ID, if the target is defined."""
        neighbors_df = self.neighbors(query_id, include_self=False)
        if self.target and not neighbors_df.empty:
            summary_stats = neighbors_df[self.target].describe()
            return pd.DataFrame(summary_stats).transpose()
        else:
            self.log.warning(f"No target values found for neighbors of Query ID '{query_id}'.")
            return pd.DataFrame()


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

    # Test using Training and Test DataFrames
    class_spider = FeatureSpaceProximity(training_df, ["feat1", "feat2", "feat3"], id_column="ID", target="class")

    # Neighbors Test
    class_neighbors = class_spider.neighbors_bulk(test_df)
    print("\nNeighbors Bulk (Test Data):\n", class_neighbors)

    # Neighbors Test (with rows from the training data)
    class_neighbors = class_spider.neighbors_bulk(training_df[:3])
    print("\nNeighbors Bulk (Training Data):\n", class_neighbors)

    # Neighbor Test using a single query ID
    single_query_id = "id_5"
    single_query_neighbors = class_spider.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)

    # Neighbor within Radius Test
    my_radius = 0.5
    radius_query = class_spider.neighbors_bulk(test_df, radius=my_radius)
    print(f"\nNeighbors within Radius Bulk {my_radius} (Test Data):\n", radius_query)

    # Neighbor within Radius Test using a single query ID
    single_query_id = "id_5"
    single_query_neighbors = class_spider.neighbors(single_query_id, radius=my_radius)
    print(f"\nNeighbors within Radius {my_radius} Query ID:", single_query_id)
    print(single_query_neighbors)

    # Target Summary Test
    single_query_id = "id_5"
    target_summary = class_spider.target_summary(single_query_id)
    print(f"\nTarget Summary for Query ID '{single_query_id}':\n", target_summary)

    # Neighbor Indices and Distances Test
    indices, distances = class_spider.get_neighbor_indices_and_distances()
    print("\nNeighbor Indices (Training Data):\n", indices)
    print("\nNeighbor Distances (Training Data):\n", distances)
