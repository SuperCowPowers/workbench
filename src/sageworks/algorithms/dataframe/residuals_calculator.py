import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


class ResidualsCalculator(BaseEstimator, TransformerMixin):
    """
    A custom transformer for calculating residuals using cross-validation.

    This transformer performs K-Fold cross-validation, generates predictions, computes residuals,
    and adds 'prediction', 'residuals', 'residuals_abs', 'residuals_100', and 'residuals_100_abs' columns to the input DataFrame.

    Attributes:
        n_splits (int): Number of splits for cross-validation.
        random_state (int): Random state for reproducibility.
        model (XGBRegressor): The machine learning model used for predictions.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initializes the ResidualsCalculator with the specified parameters.

        Args:
            n_splits (int): Number of splits for cross-validation.
            random_state (int): Random state for reproducibility.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = XGBRegressor()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the model. In this case, fitting involves storing the input data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            self: Returns an instance of self.
        """
        self.X = X
        self.y = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding 'prediction', 'residuals', 'residuals_abs', 'residuals_100', and 'residuals_100_abs' columns.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """
        if self.y is None:
            raise ValueError("Target variable 'y' is not provided. Please call 'fit' with both 'X' and 'y'.")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        prediction_list = []
        residuals_list = []
        residuals_abs_list = []

        # Perform cross-validation and collect predictions and residuals
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Fit the model on the training data
            self.model.fit(X_train, y_train)
            # Predict on the test data
            y_pred = self.model.predict(X_test)

            # Compute residuals and absolute residuals
            residuals = y_test - y_pred
            residuals_abs = np.abs(residuals)

            # Collect predictions and residuals
            prediction_list.extend(y_pred)
            residuals_list.extend(residuals)
            residuals_abs_list.extend(residuals_abs)

        # Create a copy of the input DataFrame and add the new columns
        result_df = self.X.copy()
        result_df['prediction'] = prediction_list
        result_df['residuals'] = residuals_list
        result_df['residuals_abs'] = residuals_abs_list

        # Train on all data and compute residuals for 100% training
        self.model.fit(self.X, self.y)
        y_pred_100 = self.model.predict(self.X)
        residuals_100 = self.y - y_pred_100
        residuals_100_abs = np.abs(residuals_100)

        result_df['residuals_100'] = residuals_100
        result_df['residuals_100_abs'] = residuals_100_abs

        return result_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> pd.DataFrame:
        """
        Fits the model and transforms the input DataFrame by adding 'prediction', 'residuals',
        'residuals_abs', 'residuals_100', and 'residuals_100_abs' columns.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.
            **fit_params: Additional fit parameters.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    """ Example usage of the ResidualsCalculator """
    from sageworks.api.feature_set import FeatureSet
    from sageworks.api.model import Model

    # Load the Abalone FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    model = Model("abalone-regression")
    target_column = model.target()
    feature_columns = model.features()

    # Initialize the ResidualsCalculator
    residuals_calculator = ResidualsCalculator(n_splits=5, random_state=42)
    result_df = residuals_calculator.fit_transform(df[feature_columns], df[target_column])

    # Print the result DataFrame
    print(result_df)

    # Now do the AQSol data
    fs = FeatureSet("aqsol_mol_descriptors")
    if not fs.exists():
        exit(0)
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    model = Model("aqsol-mol-regression")
    target_column = model.target()
    feature_columns = model.features()

    # Initialize the ResidualsCalculator
    residuals_calculator = ResidualsCalculator(n_splits=5, random_state=42)
    result_df = residuals_calculator.fit_transform(df[feature_columns], df[target_column])

    # Grab the residuals and residuals_abs columns
    residual_columns = ["residuals", "residuals_abs", "residuals_100", "residuals_100_abs"]
    residual_df = result_df[residual_columns]

    # Print the residual DataFrame
    print(residual_df)
