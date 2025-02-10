"""TargetGradients: Compute Feature Space to Target Gradients"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Workbench Imports
from workbench.utils.pandas_utils import remove_rows_with_nans


class TargetGradients(BaseEstimator, TransformerMixin):
    """
    A custom transformer for computing gradients between observations that are close in feature space.
    This transformer adds 'feature_diff', 'target_diff' and 'target_gradient' columns to the DataFrame.

    Attributes:
        n_neighbors (int): Number of neighbors to consider.
    """

    def __init__(self, n_neighbors: int = 2):
        """Initialize the TargetGradients with the specified parameters.

        Args:
            n_neighbors (int): Number of neighbors to consider (default: 2)
        """
        self.n_neighbors = n_neighbors
        self.scalar = StandardScaler()
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, algorithm="ball_tree", metric="euclidean")
        self.X = None
        self.y = None
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Standardize the features and fit the internal KNN model

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            self: Returns an instance of self.
        """

        # Drop rows with NaNs (we need to combine X and y, so that rows with NaNs are dropped together)
        df = pd.concat([X, y], axis=1)
        df = remove_rows_with_nans(df)
        X = df.drop(columns=[y.name])
        y = df[y.name]

        # Store the input data
        self.X = X
        self.y = y

        # Standardize the features
        X_norm = self.scalar.fit_transform(X)

        # Fit the KNN model
        self.knn.fit(X_norm, self.y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding 'feature_diff', 'target_diff' and 'target_gradient'  columns.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """

        # Drop rows with NaNs in the DataFrame
        X = remove_rows_with_nans(X)

        # Standardize the features
        df_norm = pd.DataFrame(self.scalar.fit_transform(X), columns=X.columns, index=X.index)

        # Initialize arrays to accumulate results
        feature_diff = []
        target_diff = []
        target_gradient = []

        # Compute the gradients
        for index, row in df_norm.iterrows():
            # Find the nearest neighbors
            distances, indices = self.knn.kneighbors([row], n_neighbors=self.n_neighbors)

            # Find the index of the nearest neighbor that is not the observation itself
            nn_index = indices[0][1] if indices[0][0] == index else indices[0][0]

            # Compute the difference in feature space and target space
            diff_feature = distances[0][1] if indices[0][0] == index else distances[0][0]
            diff_target = abs(self.y.iloc[index] - self.y.iloc[nn_index])

            # Calculate the target gradient, handling division by zero
            gradient_target = float("inf") if diff_feature == 0 else diff_target / diff_feature

            # Accumulate results
            feature_diff.append(diff_feature)
            target_diff.append(diff_target)
            target_gradient.append(gradient_target)

        # Add the accumulated results to the DataFrame
        df_norm["feature_diff"] = feature_diff
        df_norm["target_diff"] = target_diff
        df_norm["target_gradient"] = target_gradient

        return df_norm

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        """
        Fits the model and transforms the input DataFrame by adding 'feature_diff',
        'target_diff' and 'target_gradient'  columns.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.
        """
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    """Example usage of the TargetGradients class"""
    from workbench.api.feature_set import FeatureSet
    from workbench.api.model import Model

    # Load the Abalone FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    model = Model("abalone-regression")
    target_column = model.target()
    feature_columns = model.features()

    # Initialize the TargetGradients
    target_gradients = TargetGradients(n_neighbors=2)
    result_df = target_gradients.fit_transform(df[feature_columns], df[target_column])

    # Print the result DataFrame
    print(result_df)
