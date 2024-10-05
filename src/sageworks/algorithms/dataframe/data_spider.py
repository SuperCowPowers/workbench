import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import logging


class DataSpider:
    def __init__(self, df: pd.DataFrame, features: list, id_column: str, target_column: str, neighbors: int = 5, classification: bool = False):
        """DataSpider: A simple class for prediction and neighbor lookups using KNN.

        Args:
            df: Pandas DataFrame
            features: List of feature column names
            id_column: Name of the ID column
            target_column: Name of the target column
            neighbors: Number of neighbors to use in the KNN model (default: 5)
            classification: Boolean indicating if the target column is for classification (default: False)
        """
        self.log = logging.getLogger("sageworks")
        self.df = df.copy()
        self.features = features
        self.id_column = id_column
        self.target_column = target_column
        self.classification = classification

        # Use appropriate KNN model based on classification or regression
        if classification:
            self.knn_model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
        else:
            self.knn_model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")

        # Standardize the feature values and build the pipeline
        self.knn_pipeline = make_pipeline(StandardScaler(), self.knn_model)
        self.knn_pipeline.fit(df[features], df[target_column])  # Fit with features and the actual target column

        # Store the scaler separately for custom neighbor querying
        self.scaler = self.knn_pipeline[0]

    def predict(self, query_df: pd.DataFrame):
        """Return predictions for the given query dataframe.

        Args:
            query_df: DataFrame with the same feature columns as training data.

        Returns:
            List of predicted target values.
        """
        return self.knn_pipeline.predict(query_df[self.features])

    def predict_proba(self, query_df: pd.DataFrame):
        """Return class probabilities for classification tasks.

        Args:
            query_df: DataFrame with the same feature columns as training data.

        Returns:
            Probability distributions over classes for each query point.
        """
        if self.classification:
            return self.knn_pipeline.predict_proba(query_df[self.features])
        else:
            self.log.warning("predict_proba is only available for classification models.")
            return None

    def get_neighbors(self, query_df: pd.DataFrame):
        """Return neighbors for the given query dataframe.

        Args:
            query_df: DataFrame with the same feature columns as training data.

        Returns:
            DataFrame with neighbor IDs, target values, and distances.
        """
        # Transform the query data using the same scaler
        query_scaled = self.scaler.transform(query_df[self.features])

        # Get neighbors using the KNN internals directly
        distances, indices = self.knn_model.kneighbors(query_scaled)

        # Collect neighbor info
        query_ids = query_df[self.id_column].values
        neighbor_ids = [[self.df.iloc[idx][self.id_column] for idx in index_list] for index_list in indices]
        neighbor_targets = [[self.df.iloc[idx][self.target_column] for idx in index_list] for index_list in indices]
        neighbor_distances = [list(dist_list) for dist_list in distances]

        # Remove the query itself from the neighbor results if it appears
        for i, query_id in enumerate(query_ids):
            if query_id in neighbor_ids[i]:
                self.log.info(f"Query ID '{query_id}' found in neighbors. Removing...")
                idx_to_remove = neighbor_ids[i].index(query_id)
                neighbor_ids[i].pop(idx_to_remove)
                neighbor_targets[i].pop(idx_to_remove)
                neighbor_distances[i].pop(idx_to_remove)

        # Create and return a results DataFrame
        result_df = pd.DataFrame({
            "query_id": query_ids,
            "neighbor_ids": neighbor_ids,
            "neighbor_targets": neighbor_targets,
            "neighbor_distances": neighbor_distances,
        })
        return result_df


# Testing the DataSpider class with both regression and classification targets
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Generate a 20-row test dataset
    data = {
        "ID": [f"id_{i}" for i in range(20)],
        "feat1": [1.0, 1.1, 1.2, 3.0, 4.0, 1.0, 1.2, 3.1, 4.2, 4.3, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.3, 3.2, 4.3, 4.4, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3, 0.5, 0.4, 0.5, 0.6],
        "feat3": [0.1, 0.2, 0.3, 1.6, 2.5, 0.2, 0.3, 1.7, 2.6, 2.7, 1.0, 1.1, 1.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.5],
        "target": [10, 11, 12, 20, 30, 10, 12, 21, 31, 32, 15, 16, 17, 5, 6, 7, 8, 9, 10, 11],
        "class": [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    df = pd.DataFrame(data)

    # Regression Test
    reg_spider = DataSpider(df, ["feat1", "feat2", "feat3"], id_column="ID", target_column="target", classification=False)
    reg_predictions = reg_spider.predict(df.iloc[:2])
    print("Regression Predictions:\n", reg_predictions)

    # Classification Test
    class_spider = DataSpider(df, ["feat1", "feat2", "feat3"], id_column="ID", target_column="class", classification=True)
    class_predictions = class_spider.predict(df.iloc[:2])
    class_probs = class_spider.predict_proba(df.iloc[:2])
    print("\nClassification Predictions:\n", class_predictions)
    print("Classification Probabilities:\n", class_probs)
