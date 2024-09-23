from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted


class ResidualsCalculator(BaseEstimator, TransformerMixin):
    """
    A custom transformer for calculating residuals using cross-validation.

    This transformer performs K-Fold cross-validation, generates predictions, computes residuals,
    and adds 'prediction', 'residuals', 'residuals_abs', 'residuals_100', and 'residuals_100_abs'
    columns to the input DataFrame.

    Attributes:
        model_class (Union[RegressorMixin, XGBRegressor]): The machine learning model class used for predictions.
        n_splits (int): Number of splits for cross-validation.
        random_state (int): Random state for reproducibility.
    """

    def __init__(
        self, model: Union[RegressorMixin, XGBRegressor] = XGBRegressor, n_splits: int = 5, random_state: int = 42
    ):
        """
        Initializes the ResidualsCalculator with the specified parameters.

        Args:
            model (Union[RegressorMixin, XGBRegressor]): The machine learning model class used for predictions.
            n_splits (int): Number of splits for cross-validation.
            random_state (int): Random state for reproducibility.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_class = model  # Store the class, instantiate the model later
        self.model = None  # Lazy model initialization
        self.X = None
        self.y = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Fits the model. In this case, fitting involves storing the input data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.

        Returns:
            self: Returns an instance of self.
        """
        self.X = X
        self.y = y
        self.model = self.model_class()  # Initialize the model when fit is called
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding 'prediction', 'residuals', 'residuals_abs',
        'prediction_100', 'residuals_100', and 'residuals_100_abs' columns.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """
        check_is_fitted(self, ['X', 'y'])  # Ensure fit has been called

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Initialize pandas Series to store predictions and residuals, aligned by index
        predictions = pd.Series(index=self.y.index, dtype=np.float64)
        residuals = pd.Series(index=self.y.index, dtype=np.float64)
        residuals_abs = pd.Series(index=self.y.index, dtype=np.float64)

        # Perform cross-validation and collect predictions and residuals
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Fit the model on the training data
            self.model.fit(X_train, y_train)

            # Predict on the test data
            y_pred = self.model.predict(X_test)

            # Compute residuals and absolute residuals
            residuals_fold = y_test - y_pred
            residuals_abs_fold = np.abs(residuals_fold)

            # Place the predictions and residuals in the correct positions based on index
            predictions.iloc[test_index] = y_pred
            residuals.iloc[test_index] = residuals_fold
            residuals_abs.iloc[test_index] = residuals_abs_fold

        # Create a copy of the provided DataFrame and add the new columns
        result_df = X.copy()  # Create a copy to avoid modifying input X directly
        result_df["prediction"] = predictions
        result_df["residuals"] = residuals
        result_df["residuals_abs"] = residuals_abs

        # Train on all data and compute residuals for 100% training
        self.model.fit(self.X, self.y)
        y_pred_100 = self.model.predict(self.X)
        residuals_100 = self.y - y_pred_100
        residuals_100_abs = np.abs(residuals_100)

        result_df["prediction_100"] = y_pred_100
        result_df["residuals_100"] = residuals_100
        result_df["residuals_100_abs"] = residuals_100_abs

        return result_df


if __name__ == "__main__":
    """Example usage of the ResidualsCalculator"""
    from sageworks.api.feature_set import FeatureSet
    from sageworks.api.model import Model

    # Now do the AQSol data (with computed molecular descriptors)
    fs = FeatureSet("aqsol_mol_descriptors")
    if not fs.exists():
        exit(0)
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    aqsol_model = Model("aqsol-mol-regression")
    target_column = aqsol_model.target()
    feature_columns = aqsol_model.features()

    # Initialize the ResidualsCalculator
    residuals_calculator = ResidualsCalculator(n_splits=5, random_state=42)
    result_df = residuals_calculator.fit_transform(df[feature_columns], df[target_column])

    # Add the target column back to the result DataFrame
    result_df[target_column] = df[target_column]

    # Grab the residuals and residuals_abs columns
    residual_columns = ["residuals", "residuals_abs", "residuals_100", "residuals_100_abs"]
    residual_df = result_df[residual_columns]

    # Compute percentage of observations with residuals_100_abs > residuals_abs
    percentage = (result_df["residuals_100_abs"] > result_df["residuals_abs"]).mean()
    print(f"Percentage of observations with residuals_100_abs > residuals_abs: {percentage:.2f}")

    # Print the residual DataFrame
    print(residual_df)

    # Show a scatter plot of the residuals
    from sageworks.web_components.plugins.scatter_plot import ScatterPlot
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Columns of Interest
    dropdown_columns = ["residuals_abs", "residuals_100_abs", "prediction", "prediction_100", "solubility"]

    # Run the Unit Test on the Plugin
    unit_test = PluginUnitTest(
        ScatterPlot,
        input_data=result_df[dropdown_columns],
        x="solubility",
        y="prediction",
        color="residuals_abs",
        dropdown_columns=dropdown_columns,
    )
    unit_test.run()