from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted


class ResidualsCalculator(BaseEstimator, TransformerMixin):
    """
    A custom transformer for calculating residuals using cross-validation or an endpoint.

    This transformer performs K-Fold cross-validation (if no endpoint is provided), or it uses the endpoint
    to generate predictions and compute residuals. It adds 'prediction', 'residuals', 'residuals_abs',
    'prediction_100', 'residuals_100', and 'residuals_100_abs' columns to the input DataFrame.

    Attributes:
        model_class (Union[RegressorMixin, XGBRegressor]): The machine learning model class used for predictions.
        n_splits (int): Number of splits for cross-validation.
        random_state (int): Random state for reproducibility.
        endpoint (Optional): The SageWorks endpoint object for running inference, if provided.
    """

    def __init__(
        self,
        endpoint: Optional[object] = None,
        reference_model_class: Union[RegressorMixin, XGBRegressor] = XGBRegressor,
    ):
        """
        Initializes the ResidualsCalculator with the specified parameters.

        Args:
            endpoint (Optional): A SageWorks endpoint object to run inference, if available.
            reference_model_class (Union[RegressorMixin, XGBRegressor]): The reference model class for predictions.
        """
        self.n_splits = 5
        self.random_state = 42
        self.reference_model_class = reference_model_class  # Store the class, instantiate the model later
        self.reference_model = None  # Lazy model initialization
        self.endpoint = endpoint  # Use this endpoint for inference if provided
        self.X = None
        self.y = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Fits the model. If no endpoint is provided, fitting involves storing the input data
        and initializing a reference model.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.

        Returns:
            self: Returns an instance of self.
        """
        self.X = X
        self.y = y

        if self.endpoint is None:
            # Only initialize the reference model if no endpoint is provided
            self.reference_model = self.reference_model_class()
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
        check_is_fitted(self, ["X", "y"])  # Ensure fit has been called

        if self.endpoint:
            # If an endpoint is provided, run inference on the full data
            result_df = self._run_inference_via_endpoint(X)
        else:
            # If no endpoint, perform cross-validation and full model fitting
            result_df = self._run_cross_validation(X)

        return result_df

    def _run_cross_validation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the cross-validation process when no endpoint is provided.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: DataFrame with predictions and residuals from cross-validation and full model fit.
        """
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
            self.reference_model.fit(X_train, y_train)

            # Predict on the test data
            y_pred = self.reference_model.predict(X_test)

            # Compute residuals and absolute residuals
            residuals_fold = y_test - y_pred
            residuals_abs_fold = np.abs(residuals_fold)

            # Place the predictions and residuals in the correct positions based on index
            predictions.iloc[test_index] = y_pred
            residuals.iloc[test_index] = residuals_fold
            residuals_abs.iloc[test_index] = residuals_abs_fold

        # Train on all data and compute residuals for 100% training
        self.reference_model.fit(self.X, self.y)
        y_pred_100 = self.reference_model.predict(self.X)
        residuals_100 = self.y - y_pred_100
        residuals_100_abs = np.abs(residuals_100)

        # Create a copy of the provided DataFrame and add the new columns
        result_df = X.copy()
        result_df["prediction"] = predictions
        result_df["residuals"] = residuals
        result_df["residuals_abs"] = residuals_abs
        result_df["prediction_100"] = y_pred_100
        result_df["residuals_100"] = residuals_100
        result_df["residuals_100_abs"] = residuals_100_abs
        result_df[self.y.name] = self.y  # Add the target column back

        return result_df

    def _run_inference_via_endpoint(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the inference process when an endpoint is provided.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: DataFrame with predictions and residuals from the endpoint.
        """
        # Run inference on all data using the endpoint (include the target column)
        X = X.copy()
        X.loc[:, self.y.name] = self.y
        results_df = self.endpoint.inference(X)
        predictions = results_df["prediction"]

        # Compute residuals and residuals_abs based on the endpoint's predictions
        residuals = self.y - predictions
        residuals_abs = np.abs(residuals)

        # To maintain consistency, populate both 'prediction' and 'prediction_100' with the same values
        result_df = X.copy()
        result_df["prediction"] = predictions
        result_df["residuals"] = residuals
        result_df["residuals_abs"] = residuals_abs
        result_df["prediction_100"] = predictions
        result_df["residuals_100"] = residuals
        result_df["residuals_100_abs"] = residuals_abs

        return result_df


if __name__ == "__main__":
    """Example usage of the ResidualsCalculator"""
    from sageworks.api.feature_set import FeatureSet
    from sageworks.api.model import Model
    from sageworks.api.endpoint import Endpoint

    # Now do the AQSol data (with computed molecular descriptors)
    fs = FeatureSet("aqsol_mol_descriptors")
    if not fs.exists():
        exit(0)
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    aqsol_model = Model("aqsol-mol-regression")
    target_column = aqsol_model.target()
    feature_columns = aqsol_model.features()

    # Case 1: Use the reference model (no endpoint)
    residuals_calculator = ResidualsCalculator()
    my_result_df = residuals_calculator.fit_transform(df[feature_columns], df[target_column])

    # Case 2: Use an existing endpoint for inference
    endpoint = Endpoint("aqsol-regression-end")
    residuals_calculator_endpoint = ResidualsCalculator(endpoint=endpoint)
    result_df_endpoint = residuals_calculator_endpoint.fit_transform(df[feature_columns], df[target_column])

    # Show a scatter plot of the residuals
    from sageworks.web_components.plugins.scatter_plot import ScatterPlot
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Columns of Interest
    dropdown_columns = ["residuals", "residuals_abs", "prediction", "solubility"]

    # Run the Unit Test on the Plugin
    unit_test = PluginUnitTest(
        ScatterPlot,
        input_data=result_df_endpoint[dropdown_columns],
        x="solubility",
        y="prediction",
        color="residuals_abs",
        dropdown_columns=dropdown_columns,
    )
    unit_test.run()
