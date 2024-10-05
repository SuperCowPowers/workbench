import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DataSpider:
    def __init__(self, df: pd.DataFrame, features: list, id_column: str, target_column: str, neighbors: int = 5):
        """DataSpider: A simple class for prediction and neighbor lookups using KNN.

        Args:
            df: Pandas DataFrame
            features: List of feature column names
            id_column: Name of the ID column
            target_column: Name of the target column
            neighbors: Number of neighbors to use in the KNN model (default: 5)
        """
        self.df = df.copy()
        self.features = features
        self.id_column = id_column
        self.target_column = target_column

        # Standardize the feature values and build the KNN regression model
        self.knn_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=neighbors, weights="distance"))
        self.knn_pipeline.fit(df[features], df[target_column])  # Fit with features and the actual target column

        # Store the scaler and the KNN model separately for custom neighbor querying
        self.scaler = self.knn_pipeline[0]
        self.knn_model = self.knn_pipeline[-1]

    def predict(self, query_df: pd.DataFrame):
        """Return predictions for the given query dataframe.

        Args:
            query_df: DataFrame with the same feature columns as training data.

        Returns:
            List of predicted target values.
        """
        return self.knn_pipeline.predict(query_df[self.features])

    def get_neighbors(self, query_df: pd.DataFrame):
        """Return neighbors for the given query dataframe.

        Args:
            query_df: DataFrame with the same feature columns as training data.

        Returns:
            DataFrame with neighbor IDs, target values, and distances.
        """
        # Transform the query data using the same scaler
        query_scaled = self.scaler.transform(query_df[self.features])

        # Get neighbors using the KNeighborsRegressor's internals directly
        distances, indices = self.knn_model.kneighbors(query_scaled)

        # Collect neighbor info
        neighbor_ids = [[self.df.iloc[idx][self.id_column] for idx in index_list] for index_list in indices]
        neighbor_targets = [[self.df.iloc[idx][self.target_column] for idx in index_list] for index_list in indices]
        neighbor_distances = [list(dist_list) for dist_list in distances]

        # Create a results DataFrame
        result_df = pd.DataFrame({
            "query_id": query_df[self.id_column].values,
            "neighbor_ids": neighbor_ids,
            "neighbor_targets": neighbor_targets,
            "neighbor_distances": neighbor_distances,
        })
        return result_df


# Testing the DataSpider class
if __name__ == "__main__":

    # Change pandas display settings for better readability
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    
    # Generate a 20-row test dataset
    data = {
        "ID": [f"id_{i}" for i in range(20)],
        "feat1": [1.0, 1.1, 1.2, 3.0, 4.0, 1.0, 1.2, 3.1, 4.2, 4.3, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.3, 3.2, 4.3, 4.4, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3, 0.5, 0.4, 0.5, 0.6],
        "feat3": [0.1, 0.2, 0.3, 1.6, 2.5, 0.2, 0.3, 1.7, 2.6, 2.7, 1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.5],
        "target": [10, 11, 12, 20, 30, 10, 12, 21, 31, 32, 15, 16, 17, 5, 6, 7, 8, 9, 10, 11],
    }
    df = pd.DataFrame(data)

    # Create DataSpider instance
    data_spider = DataSpider(df, ["feat1", "feat2", "feat3"], id_column="ID", target_column="target")

    # Predict on a sample query point
    query_df = df[df["ID"] == "id_0"]
    predictions = data_spider.predict(query_df)
    print(f"Prediction for query point 'id_0': {predictions}")

    # Query for neighbors of a specific point
    neighbors = data_spider.get_neighbors(query_df)
    print("\nNeighbors for query point:\n", neighbors)

    # Query for a different point to check consistency
    query_df2 = df[df["ID"] == "id_10"]
    neighbors2 = data_spider.get_neighbors(query_df2)
    print("\nNeighbors for second query point:\n", neighbors2)