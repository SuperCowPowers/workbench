from typing import Union
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import logging

# SageWorks Imports
from sageworks.api import FeatureSet, Model


class KNNSpiderDeprecated:
    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        target: str,
        id_column: str,
        neighbors: int = 10,
        classification: bool = False,
        class_labels: list = None,
    ):
        """KNNSpiderDeprecated: A simple class for prediction and neighbor lookups using KNN.

        Args:
            df: Pandas DataFrame
            features: List of feature column names
            target: Name of the target column
            id_column: Name of the ID column
            neighbors: Number of neighbors to use in the KNN model (default: 10)
            classification: Boolean indicating if the target column is for classification (default: False)
            class_labels: Optional list of class labels in the desired order (e.g., ["low", "medium", "high"])
        """
        self.log = logging.getLogger("sageworks")
        self.df = df.copy()
        self.features = features
        self.id_column = id_column
        self.target = target
        self.classification = classification
        self.class_labels = class_labels

        # Use appropriate KNN model based on classification or regression
        if classification:
            self.knn_model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
        else:
            self.knn_model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")

        # Standardize the feature values and build the pipeline
        self.knn_pipeline = make_pipeline(StandardScaler(), self.knn_model)
        self.knn_pipeline.fit(df[features], df[target])  # Fit with features and the actual target column

        # Store the scaler separately for custom neighbor querying
        self.scaler = self.knn_pipeline[0]

        # If using classification and class_labels are provided, reorder classes
        if self.classification and self.class_labels:
            self._reorder_class_labels()

        # Store the internally fitted feature matrix after scaling
        self.scaled_X = self.knn_model._fit_X

    @classmethod
    def from_model(cls, model: Model, id_column: str):
        """Create a KNNSpiderDeprecated instance from a SageWorks model object

        Args:
            model (Model): A SageWorks model object
            id_column (str): Name of the ID column

        Returns:
            KNNSpiderDeprecated: A new instance of the KNNSpiderDeprecated class
        """

        # Extract necessary information from the SageWorks model
        fs = FeatureSet(model.get_input())
        df = fs.view("training").pull_dataframe()
        features = model.features()
        target = model.target()
        classification = model.model_type.value == "classification"
        if classification:
            class_labels = model.class_labels()
        else:
            class_labels = None

        # Pass the extracted arguments to the existing __init__ method
        return cls(
            df,
            features=features,
            target=target,
            id_column=id_column,
            classification=classification,
            class_labels=class_labels,
        )

    def get_neighbor_indices_and_distances(self):
        """Retrieve neighbor indices and distances for all points in the dataset."""
        # Use the already scaled feature matrix stored in `self.scaled_X`
        distances, indices = self.knn_model.kneighbors(self.scaled_X)
        return indices, distances

    def predict(self, query_df: pd.DataFrame):
        """Return predictions for the given query dataframe."""
        return self.knn_pipeline.predict(query_df[self.features])

    def predict_proba(self, query_df: pd.DataFrame):
        """Return class probabilities for classification tasks."""
        if self.classification:
            probas = self.knn_pipeline.predict_proba(query_df[self.features])
            if hasattr(self, "class_reorder_map"):
                reordered_probas = probas[:, self.class_reorder_map]
                return reordered_probas
            return probas
        else:
            self.log.warning("predict_proba is only available for classification models.")
            return None

    def neighbors(self, query_id: Union[str, int], include_self: bool = True) -> pd.DataFrame:
        """Return a filtered DataFrame with neighbors of the given query ID.

        Args:
            query_id (Union[str, int]): The ID of the query point.
            include_self (bool): Whether to include the query ID itself in the neighbor results.

        Returns:
            pd.DataFrame: Filtered DataFrame that includes the query ID and its neighbors.
        """
        # Check if the query ID is in the internal DataFrame
        if query_id not in self.df[self.id_column].values:
            self.log.warning(f"Query ID '{query_id}' not found in the DataFrame. Returning an empty DataFrame.")
            return pd.DataFrame()

        # Get a single-row DataFrame for the query ID
        query_df = self.df[self.df[self.id_column] == query_id]

        # Use the existing method to get neighbors for this single query row
        neighbors_info_df = self.neighbors_bulk(query_df, include_self)

        # Extract the neighbor IDs and distances from the results
        neighbor_ids = neighbors_info_df["neighbor_ids"].iloc[0]  # List of neighbor IDs
        neighbor_distances = neighbors_info_df["neighbor_distances"].iloc[0]  # Corresponding distances

        # Filter the internal DataFrame to include only the neighbors
        neighbors_df = self.df[self.df[self.id_column].isin(neighbor_ids)]

        # Reindex to ensure the rows are in the same order as `neighbor_ids`
        neighbors_df = neighbors_df.set_index(self.id_column).reindex(neighbor_ids).reset_index()

        # Attach the corresponding distances to the filtered DataFrame
        neighbors_df["knn_distance"] = neighbor_distances

        return neighbors_df

    def neighbors_bulk(self, query_df: pd.DataFrame, include_self: bool = False) -> pd.DataFrame:
        """Return all the neighbors for each row in the given query dataframe.

        Args:
            query_df: Pandas DataFrame with the same features as the training data.
            include_self: Boolean indicating whether to include the query ID in the neighbor results.

        Returns:
            pd.DataFrame: DataFrame with query ID, neighbor IDs, neighbor targets, and neighbor distances.
        """

        # Scale the query data using the same scaler as the training data
        query_scaled = self.scaler.transform(query_df[self.features])

        # Retrieve neighbors and distances using the KNN internals
        distances, indices = self.knn_model.kneighbors(query_scaled)

        # Collect neighbor information (IDs, target values, and distances)
        query_ids = query_df[self.id_column].values
        neighbor_ids = [[self.df.iloc[idx][self.id_column] for idx in index_list] for index_list in indices]
        neighbor_targets = [[self.df.iloc[idx][self.target] for idx in index_list] for index_list in indices]
        neighbor_distances = [list(dist_list) for dist_list in distances]

        # Automatically remove the query ID itself from the neighbor results
        for i, query_id in enumerate(query_ids):
            if query_id in neighbor_ids[i] and not include_self:
                idx_to_remove = neighbor_ids[i].index(query_id)
                neighbor_ids[i].pop(idx_to_remove)
                neighbor_targets[i].pop(idx_to_remove)
                neighbor_distances[i].pop(idx_to_remove)

        # Create and return a results DataFrame with the updated neighbor information
        result_df = pd.DataFrame(
            {
                "query_id": query_ids,
                "neighbor_ids": neighbor_ids,
                "neighbor_targets": neighbor_targets,
                "neighbor_distances": neighbor_distances,
            }
        )
        return result_df

    def _reorder_class_labels(self):
        """Reorder the class labels based on the specified class_labels order."""
        original_labels = list(self.knn_model.classes_)
        if not set(self.class_labels) <= set(original_labels):
            self.log.warning(
                f"Some class labels {self.class_labels} are missing from the model's fitted classes: {original_labels}."
            )
            self.class_labels = original_labels
        self.class_reorder_map = [original_labels.index(label) for label in self.class_labels]


# Testing the KNNSpiderDeprecated class with separate training and test/evaluation dataframes
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
    reg_spider = KNNSpiderDeprecated(
        training_df, ["feat1", "feat2", "feat3"], target="target", id_column="ID", classification=False
    )
    reg_predictions = reg_spider.predict(test_df)
    print("Regression Predictions (Test Data):\n", reg_predictions)

    # Regression Neighbors Test
    reg_neighbors = reg_spider.neighbors_bulk(test_df)
    print("\nRegression Neighbors (Test Data):\n", reg_neighbors)

    # Classification Test using Training and Test DataFrames
    class_spider = KNNSpiderDeprecated(
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

    # Test the `from_model` class method
    model = Model("abalone-regression")
    spider_from_model = KNNSpiderDeprecated.from_model(model, id_column="id")

    # Test the `from_model` class method
    model = Model("wine-classification")
    spider_from_model = KNNSpiderDeprecated.from_model(model, id_column="id")
