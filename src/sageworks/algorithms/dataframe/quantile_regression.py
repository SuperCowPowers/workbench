from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor


class QuantileRegressor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for calculating residuals using cross-validation.

    This transformer performs K-Fold cross-validation, generates predictions, computes residuals,
    and adds 'prediction', 'residuals', 'residuals_abs', 'residuals_100', and 'residuals_100_abs'
    columns to the input DataFrame.
    """

    def __init__(self, model: Union[RegressorMixin, XGBRegressor] = XGBRegressor):
        """
        Initializes the QuantileRegressor with the specified parameters.

        Args:
            model (Union[RegressorMixin, XGBRegressor]): The machine learning model used for predictions.
        """
        self.model_factory = model
        self.models = {}
        self.quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Fits the model. In this case, fitting involves storing the input data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            self: Returns an instance of self.
        """
        """
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.5,  # Adjust as needed for different quantiles
            "n_estimators": 50,     # Fewer trees for less refinement
            "max_depth": 3,         # Shallow trees
            "learning_rate": 0.1,   # Lower learning rate
            "subsample": 0.8,       # Subsample data to introduce randomness
            "colsample_bytree": 0.8 # Subsample features
        }
        """
        # Train models for each of the quantiles
        for q in self.quantiles:
            params = {
                "objective": "reg:quantileerror",
                "eval_metric": "mae",
                "quantile_alpha": q,
                "n_estimators": 400,  # Many estimators
                # "max_depth": 1,  # Shallow trees
            }
            model = self.model_factory(**params)
            model.fit(X, y)

            # Store the model
            self.models[q] = model

        # Return the instance of self (for method chaining)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding 'quantile_05', 'quantile_50', and 'quantile_95' columns.

        Args:
            X (pd.DataFrame): The input features for the confidence model.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """

        # Run predictions for each quantile
        quantile_predictions = {q: self.models[q].predict(X) for q in self.quantiles}

        # Create a copy of the provided DataFrame and add the new columns
        result_df = X.copy()
        result_df["quantile_05"] = quantile_predictions[self.quantiles[0]]
        result_df["quantile_25"] = quantile_predictions[self.quantiles[1]]
        result_df["quantile_50"] = quantile_predictions[self.quantiles[2]]
        result_df["quantile_75"] = quantile_predictions[self.quantiles[3]]
        result_df["quantile_95"] = quantile_predictions[self.quantiles[4]]

        # Return the transformed DataFrame
        return result_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.DataFrame:
        """
        Fits the model and transforms the input DataFrame by adding 'quantile_05', 'quantile_50',
        and 'quantile_95' columns.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.
            **fit_params: Additional fit parameters.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """
        self.fit(X, y)
        return self.transform(X)


def unit_test():
    """Unit test for the QuantileRegressor"""
    from sageworks.utils.test_data_generator import TestDataGenerator
    from sageworks.web_components.plugins.scatter_plot import ScatterPlot
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Generate some random data
    generator = TestDataGenerator()
    df = generator.regression_with_varying_noise(n_samples=1000, n_features=1)

    # Grab the target and feature columns
    target_column = "target"
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]

    # Initialize the Confidence Model (QuantileRegressor)
    residuals_calculator = QuantileRegressor()

    # Fit the confidence model with all the data
    confidence_df = residuals_calculator.fit_transform(X, y)
    confidence_df[target_column] = y

    # Compute the intervals
    confidence_df["interval"] = confidence_df["quantile_95"] - confidence_df["quantile_05"]

    # Columns of Interest
    dropdown_columns = [
        "quantile_05",
        "quantile_25",
        "quantile_50",
        "quantile_75",
        "quantile_95",
        "interval",
        target_column,
    ]

    # Run the Unit Test on the Plugin
    plugin_test = PluginUnitTest(
        ScatterPlot,
        input_data=confidence_df[dropdown_columns],
        x=target_column,
        y="quantile_50",
        color="interval",
        dropdown_columns=dropdown_columns,
    )
    plugin_test.run()


def integration_test():
    from sageworks.api.feature_set import FeatureSet
    from sageworks.api.model import Model
    from sageworks.web_components.plugins.scatter_plot import ScatterPlot
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Load the AQSol data (with given features)
    fs = FeatureSet("aqsol_features")
    # fs = FeatureSet("aqsol_mol_descriptors")
    if not fs.exists():
        exit(0)
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    model = Model("aqsol-regression")
    # model = Model("aqsol-mol-regression")
    target_column = model.target()
    feature_columns = model.features()

    X = df[feature_columns]
    y = df[target_column]

    # Initialize the Confidence Model (QuantileRegressor)
    residuals_calculator = QuantileRegressor()

    # Fit the confidence model with all the data
    confidence_df = residuals_calculator.fit_transform(X, y)
    confidence_df[target_column] = y

    # Compute the intervals
    confidence_df["interval"] = confidence_df["quantile_95"] - confidence_df["quantile_05"]

    # Confidence is domain specific (in this case any interval > 4 logS unit is considered low confidence)
    confidence_df["confidence"] = 1.0 - (np.clip(confidence_df["interval"], 0, 4) * 0.25)

    # Columns of Interest
    dropdown_columns = [
        "quantile_05",
        "quantile_25",
        "quantile_50",
        "quantile_75",
        "quantile_95",
        "interval",
        "confidence",
        target_column,
    ]

    # Run the Unit Test on the Plugin
    plugin_test = PluginUnitTest(
        ScatterPlot,
        input_data=confidence_df[dropdown_columns],
        x=target_column,
        y="quantile_50",
        color="confidence",
        dropdown_columns=dropdown_columns,
    )
    plugin_test.run()


if __name__ == "__main__":
    """Example usage of the QuantileRegressor"""

    # Run the tests
    # unit_test()
    integration_test()
