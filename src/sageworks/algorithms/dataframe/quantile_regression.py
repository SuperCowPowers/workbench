from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor


class QuantileRegressor(BaseEstimator, TransformerMixin):
    """
    A class for training regression models over a set of quantiles. Useful for calculating confidence intervals.
    """

    def __init__(
        self,
        model: Union[RegressorMixin, XGBRegressor] = XGBRegressor,
        quantiles: list = [0.05, 0.25, 0.50, 0.75, 0.95],
    ):
        """
        Initializes the QuantileRegressor with the specified parameters.

        Args:
            model (Union[RegressorMixin, XGBRegressor]): The machine learning model used for predictions.
            quantiles (list): The quantiles to calculate (default: [0.05, 0.25, 0.50, 0.75, 0.95]).
        """
        self.model_factory = model
        self.q_models = {}
        self.quantiles = quantiles
        self.rmse_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Fits the model. In this case, fitting involves storing the input data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            self: Returns an instance of self.
        """

        # Train models for each of the quantiles
        for q in self.quantiles:
            params = {
                "objective": "reg:quantileerror",
                "quantile_alpha": q,
                "n_estimators": 100,  # Fewer estimators
                "max_depth": 1,  # Shallow trees
            }
            model = self.model_factory(**params)
            model.fit(X, y)

            # Store the model
            self.q_models[q] = model

        # Train a model for RMSE predictions
        params = {"objective": "reg:squarederror"}
        self.rmse_model = self.model_factory(**params)
        self.rmse_model.fit(X, y)

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
        quantile_predictions = {q: self.q_models[q].predict(X) for q in self.quantiles}

        # Create a copy of the provided DataFrame and add the new columns
        result_df = X.copy()

        # Add the quantile predictions to the DataFrame
        for q in self.quantiles:
            result_df[f"q_{int(q*100):02}"] = quantile_predictions[q]

        # Add the RMSE predictions to the DataFrame
        result_df["prediction"] = self.rmse_model.predict(X)

        # Return the transformed DataFrame
        return result_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.DataFrame:
        """
        Fits the model and transforms the input DataFrame by adding quantile columns and a prediction column.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.
            **fit_params: Additional fit parameters.

        Returns:
            pd.DataFrame: The transformed DataFrame with additional columns.
        """
        self.fit(X, y)
        return self.transform(X)


# Calculate confidence based on the quantile predictions
def example_confidence(q_dataframe, target="target", target_sensitivity=0.25):
    lower_05 = q_dataframe["q_05"]
    lower_25 = q_dataframe["q_25"]
    quant_50 = q_dataframe["q_50"]
    upper_75 = q_dataframe["q_75"]
    upper_95 = q_dataframe["q_95"]
    y = q_dataframe[target]

    # Domain specific logic for calculating confidence
    # If the interval with is greater than target_sensitivity with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    confidence_interval = upper_95 - lower_05
    q_conf = np.clip(1 - confidence_interval / (target_sensitivity * 4.0), 0, 1)

    # Now lets look at the IQR distance for each observation
    epsilon_iqr = target_sensitivity * 0.5
    iqr = np.maximum(epsilon_iqr, np.abs(upper_75 - lower_25))
    iqr_distance = np.abs(y - quant_50) / iqr
    iqr_conf = np.clip(1 - iqr_distance, 0, 1)

    # Now combine the two confidence values
    confidence = (q_conf + iqr_conf) / 2
    return confidence, q_conf, iqr_conf


def unit_test():
    """Unit test for the QuantileRegressor"""
    from sageworks.utils.test_data_generator import TestDataGenerator
    from sageworks.web_components.plugins.scatter_plot import ScatterPlot
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Generate some random data
    generator = TestDataGenerator()
    df = generator.confidence_data(n_samples=1000)

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
    confidence_df["interval"] = confidence_df["q_95"] - confidence_df["q_05"]

    # Compute the confidence
    confidence_df["conf"], confidence_df["q_conf"], confidence_df["iqr_conf"] = example_confidence(confidence_df)

    # Columns of Interest
    q_columns = [c for c in confidence_df.columns if c.startswith("q_")]
    dropdown_columns = q_columns + ["interval", "conf", "q_conf", "iqr_conf", "prediction", target_column]
    dropdown_columns += feature_columns

    # Run the Unit Test on the Plugin
    plugin_test = PluginUnitTest(
        ScatterPlot,
        input_data=confidence_df[dropdown_columns],
        x=feature_columns[0],
        y=target_column,
        color="conf",
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
    confidence_df["interval"] = confidence_df["q_95"] - confidence_df["q_05"]

    # Compute the confidence
    confidence_df["conf"], confidence_df["q_conf"], confidence_df["iqr_conf"] = example_confidence(
        confidence_df, target_column, target_sensitivity=1.5
    )

    # Columns of Interest
    q_columns = [c for c in confidence_df.columns if c.startswith("q_")]
    dropdown_columns = q_columns + ["interval", "conf", "q_conf", "iqr_conf", "prediction", target_column]
    dropdown_columns += feature_columns

    # Run the Unit Test on the Plugin
    plugin_test = PluginUnitTest(
        ScatterPlot,
        input_data=confidence_df[dropdown_columns],
        x="prediction",
        y="solubility",
        color="conf",
        dropdown_columns=dropdown_columns,
    )
    plugin_test.run()


if __name__ == "__main__":
    """Example usage of the QuantileRegressor"""

    # Run the tests
    unit_test()
    # integration_test()
