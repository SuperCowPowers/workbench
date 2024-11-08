"""FeatureSpaceProximity: A class for neighbor lookups using KNN with optional target information."""

from typing import Union
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import logging


class FeatureSpaceProximity:
    def __init__(self, df: pd.DataFrame, features: list, id_column: str, target: str = None, neighbors: int = 10):
        """FeatureSpaceProximity: A class for neighbor lookups using KNN with optional target information.

        Args:
            df: Pandas DataFrame
            features: List of feature column names
            id_column: Name of the ID column
            target: Optional name of the target column to include target-based functionality (default: None)
            neighbors: Number of neighbors to use in the KNN model (default: 10)
        """
        self.log = logging.getLogger("sageworks")
        self.df = df
        self.features = features
        self.id_column = id_column
        self.target = target
        self.knn_neighbors = neighbors

        # Standardize the feature values and build the KNN model
        self.log.info("Building KNN model for FeatureSpaceProximity...")
        self.scaler = StandardScaler().fit(df[features])
        scaled_features = self.scaler.transform(df[features])
        self.knn_model = NearestNeighbors(n_neighbors=neighbors, algorithm="auto").fit(scaled_features)

        # Compute Z-Scores or Consistency Scores for the target values
        if self.target and is_numeric_dtype(self.df[self.target]):
            self.log.info("Computing Z-Scores for target values...")
            self.target_z_scores()
        else:
            self.log.info("Computing target consistency scores...")
            self.target_consistency()

        # Now compute the outlier scores
        self.log.info("Computing outlier scores...")
        self.outliers()

    @classmethod
    def from_model(cls, model) -> "FeatureSpaceProximity":
        """Create a FeatureSpaceProximity instance from a SageWorks model object.

        Args:
            model (Model): A SageWorks model object.

        Returns:
            FeatureSpaceProximity: A new instance of the FeatureSpaceProximity class.
        """
        from sageworks.api import FeatureSet

        # Extract necessary attributes from the SageWorks model
        fs = FeatureSet(model.get_input())
        features = model.features()
        target = model.target()

        # Retrieve the training DataFrame from the feature set
        df = fs.view("training").pull_dataframe()

        # Create and return a new instance of FeatureSpaceProximity
        return cls(df=df, features=features, id_column=fs.id_column, target=target)

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

    def outliers(self) -> None:
        """Compute a unified 'outlier' score based on either 'target_z' or 'target_consistency'."""
        if "target_z" in self.df.columns:
            # Normalize Z-Scores to a 0-1 range
            self.df["outlier"] = (self.df["target_z"].abs() / (self.df["target_z"].abs().max() + 1e-6)).clip(0, 1)

        elif "target_consistency" in self.df.columns:
            # Calculate outlier score as 1 - consistency
            self.df["outlier"] = 1 - self.df["target_consistency"]

        else:
            self.log.warning("No 'target_z' or 'target_consistency' column found to compute outlier scores.")

    def target_z_scores(self) -> None:
        """Compute Z-Scores for NUMERIC target values."""
        if not self.target:
            self.log.warning("No target column defined for Z-Score computation.")
            return

        # Get the neighbors and distances for each internal observation
        distances, indices = self.knn_model.kneighbors()

        # Retrieve all neighbor target values in a single operation
        neighbor_targets = self.df[self.target].values[indices]  # Shape will be (n_samples, n_neighbors)

        # Compute the mean and std along the neighbors axis (axis=1)
        neighbor_means = neighbor_targets.mean(axis=1)
        neighbor_stds = neighbor_targets.std(axis=1, ddof=0)

        # Vectorized Z-score calculation
        current_targets = self.df[self.target].values
        z_scores = np.where(neighbor_stds == 0, 0.0, (current_targets - neighbor_means) / neighbor_stds)

        # Assign the computed Z-Scores back to the DataFrame
        self.df["target_z"] = z_scores

    def target_consistency(self) -> None:
        """Compute a Neighborhood Consistency Score for CATEGORICAL targets."""
        if not self.target:
            self.log.warning("No target column defined for neighborhood consistency computation.")
            return

        # Get the neighbors and distances for each internal observation (already excludes the query)
        distances, indices = self.knn_model.kneighbors()

        # Calculate the Neighborhood Consistency Score for each observation
        consistency_scores = []
        for idx, idx_list in enumerate(indices):
            query_target = self.df.iloc[idx][self.target]  # Get current observation's target value

            # Get the neighbors' target values
            neighbor_targets = self.df.iloc[idx_list][self.target]

            # Calculate the proportion of neighbors that have the same category as the query observation
            consistency_score = (neighbor_targets == query_target).mean()
            consistency_scores.append(consistency_score)

        # Add the 'target_consistency' column to the internal dataframe
        self.df["target_consistency"] = consistency_scores

    def get_neighbor_indices_and_distances(self):
        """Retrieve neighbor indices and distances for all points in the dataset."""
        distances, indices = self.knn_model.kneighbors()
        return indices, distances

    def target_summary(self, query_id: Union[str, int]) -> pd.DataFrame:
        """WIP: Provide a summary of target values in the neighborhood of the given query ID"""
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

    # Hack to set the value of the 'class' field for the first row
    # training_df.loc[0, "class"] = "high"

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

    # Test the spider using a Classification target
    spider = FeatureSpaceProximity(training_df, ["feat1", "feat2", "feat3"], id_column="ID", target="class")

    # Neighbors Bulk Test
    neighbors = spider.neighbors_bulk(test_df)
    print("\nNeighbors Bulk (Test Data):\n", neighbors)

    # Neighbors Test (with rows from the training data)
    neighbors = spider.neighbors_bulk(training_df[:3])
    print("\nNeighbors Bulk (Training Data):\n", neighbors)

    # Neighbor Test using a single query ID
    single_query_id = "id_5"
    single_query_neighbors = spider.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)

    # Neighbor within Radius Test
    my_radius = 0.5
    radius_query = spider.neighbors_bulk(test_df, radius=my_radius)
    print(f"\nNeighbors within Radius Bulk {my_radius} (Test Data):\n", radius_query)

    radius_query = spider.neighbors_bulk(test_df[2:3], radius=1.0)
    print(f"\nNeighbors within Radius Bulk {my_radius} (Test Data):\n", radius_query)

    # Neighbor within Radius Test using a single query ID
    single_query_id = "id_5"
    single_query_neighbors = spider.neighbors(single_query_id, radius=my_radius)
    print(f"\nNeighbors within Radius {my_radius} Query ID:", single_query_id)
    print(single_query_neighbors)

    # Target Summary Test
    single_query_id = "id_5"
    target_summary = spider.target_summary(single_query_id)
    print(f"\nTarget Summary for Query ID '{single_query_id}':\n", target_summary)

    # Neighbor Indices and Distances Test
    indices, distances = spider.get_neighbor_indices_and_distances()
    print("\nNeighbor Indices (Training Data):\n", indices)
    print("\nNeighbor Distances (Training Data):\n", distances)

    # Test the spider using a Regression target
    spider = FeatureSpaceProximity(training_df, ["feat1", "feat2", "feat3"], id_column="ID", target="target")

    # Create a FeatureSpaceProximity instance from a SageWorks model object
    from sageworks.api import Model

    model = Model("abalone-regression")
    model_spider = FeatureSpaceProximity.from_model(model, "id")

    # Neighbors Test using a single query ID
    single_query_id = 5
    single_query_neighbors = model_spider.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)
