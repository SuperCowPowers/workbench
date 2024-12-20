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
        super().__init__()

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
                # "n_estimators": 1000,  # Number of estimators
                # "max_depth": 1,  # Shallow trees
            }
            model = self.model_factory(**params)
            model.fit(X, y)

            # Convert quantile to string
            q_str = f"q_{int(q * 100):02}"

            # Store the model
            self.q_models[q_str] = model

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
        quantile_predictions = {q: model.predict(X) for q, model in self.q_models.items()}

        # Create a copy of the provided DataFrame and add the new columns
        result_df = X.copy()

        # Add the quantile predictions to the DataFrame
        for name, preds in quantile_predictions.items():
            result_df[name] = preds

        # Add the RMSE predictions to the DataFrame
        result_df["mean"] = self.rmse_model.predict(X)

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
def example_confidence(q_dataframe, target_sensitivity=0.25):
    lower_05 = q_dataframe["q_05"]
    lower_25 = q_dataframe["q_25"]
    quant_50 = q_dataframe["q_50"]
    upper_75 = q_dataframe["q_75"]
    upper_95 = q_dataframe["q_95"]
    y = q_dataframe["mean"]

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
    # confidence = q_conf
    return confidence, confidence_interval, iqr_distance


def example_confidence_norm(q_dataframe):
    lower_05 = q_dataframe["q_05"]
    # lower_25 = q_dataframe["q_25"]
    quant_50 = q_dataframe["q_50"]
    # upper_75 = q_dataframe["q_75"]
    upper_95 = q_dataframe["q_95"]
    y = q_dataframe["mean"]

    # Domain specific logic for calculating confidence
    # If the interval with is greater than target_sensitivity with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    interval = upper_95 - lower_05

    # Normalize the confidence interval between 0 and 1
    interval_conf = 1 - (interval - np.min(interval)) / (np.max(interval) - np.min(interval))

    # Now lets look at the mean to median distance for each observation
    mean_to_median = np.abs(y - quant_50)

    # Normalize the mean_to_median distance between 0 and 1
    mean_to_median_conf = 1 - (mean_to_median - np.min(mean_to_median)) / (
        np.max(mean_to_median) - np.min(mean_to_median)
    )

    # Now combine the two confidence values
    confidence = (interval_conf + mean_to_median_conf) / 2
    confidence *= confidence
    return confidence, interval, mean_to_median


def solubility_confidence(q_dataframe):
    lower_05 = q_dataframe["q_05"]
    # lower_25 = q_dataframe["q_25"]
    quant_50 = q_dataframe["q_50"]
    # upper_75 = q_dataframe["q_75"]
    upper_95 = q_dataframe["q_95"]
    y = q_dataframe["mean"]

    # Domain specific logic for calculating confidence
    interval = upper_95 - lower_05

    # Normalize the confidence interval between 0 and 1
    interval_conf = 1 - (interval - np.min(interval)) / (np.max(interval) - np.min(interval))

    # The boundaries are -4 and -5 for solubility
    decision_boundary_lower = -5
    decision_boundary_upper = -4

    # Element-wise condition check for proximity to decision boundaries
    close_to_boundary = (np.abs(y - decision_boundary_lower) <= 0.2) | (np.abs(y - decision_boundary_upper) <= 0.2)

    # Multiply confidence by 0.5 where the condition is met
    interval_conf[close_to_boundary] *= 0.5

    # Now let's look at the mean to median distance for each observation
    mean_to_median = np.abs(y - quant_50)

    # Normalize the mean_to_median distance between 0 and 1
    mean_to_median_conf = 1 - (mean_to_median - np.min(mean_to_median)) / (
        np.max(mean_to_median) - np.min(mean_to_median)
    )

    # Now combine the two confidence values
    confidence = (interval_conf + mean_to_median_conf) / 2
    confidence *= confidence

    return confidence, interval, mean_to_median


def confusion_matrix(confidence_df):
    from sklearn.metrics import confusion_matrix

    # Define the target boundaries
    def categorize(value):
        if value < -5:
            return "low"
        elif -5 <= value <= -4:
            return "medium"
        else:
            return "high"

    # Apply categorization to actual and predicted values
    actual_categories = confidence_df["solubility"].apply(categorize)
    predicted_categories = confidence_df["mean"].apply(categorize)

    # Compute the confusion matrix
    cm = confusion_matrix(actual_categories, predicted_categories, labels=["low", "medium", "high"])

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)


def unit_test():
    """Unit test for the QuantileRegressor"""
    from workbench.utils.test_data_generator import TestDataGenerator
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Generate some random data
    generator = TestDataGenerator()
    df = generator.confidence_data()

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
    confidence_df["conf"], confidence_df["quan_conf"], confidence_df["iqr_conf"] = example_confidence_norm(
        confidence_df
    )

    # Compute model metrics for RMSE
    rmse = np.sqrt(np.mean((confidence_df[target_column] - confidence_df["mean"]) ** 2))
    print(f"RMSE: {rmse} support: {len(confidence_df)}")

    # Domain Specific Confidence Threshold
    thres = 0.6

    # Make a 'high_confidence' column based on the threshold, casting to int (0, 1)
    confidence_df["high_confidence"] = (confidence_df["conf"] > thres).astype(int)

    # Now filter the data based on confidence and give RMSE for the filtered data
    high_confidence = confidence_df[confidence_df["high_confidence"] == 1]
    rmse = np.sqrt(np.mean((high_confidence[target_column] - high_confidence["mean"]) ** 2))
    print(f"RMSE: {rmse} support: {len(high_confidence)}")

    # Columns of Interest
    dropdown_columns = ["interval", "conf", "quan_conf", "iqr_conf", "mean", "high_confidence", target_column]
    q_columns = [c for c in confidence_df.columns if c.startswith("q_")]
    dropdown_columns += q_columns
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
    from workbench.api.feature_set import FeatureSet
    from workbench.api.model import Model
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Load the AQSol data (with given features)
    # fs = FeatureSet("aqsol_features")
    fs = FeatureSet("aqsol_mol_descriptors")
    print(f"Feature Set: {fs}")
    if not fs.exists():
        exit(0)
    df = fs.pull_dataframe()

    # Grab the target and feature columns from the model
    # model = Model("aqsol-regression")
    model = Model("aqsol-mol-regression")
    target_column = model.target()
    feature_columns = model.features()

    X = df[feature_columns]
    y = df[target_column]

    # Initialize the Confidence Model (QuantileRegressor)
    residuals_calculator = QuantileRegressor()

    # Fit the confidence model with all the data
    confidence_df = residuals_calculator.fit_transform(X, y)
    confidence_df[target_column] = y

    # Compute the confidence
    confidence_df["conf"], confidence_df["interval"], confidence_df["mean_diff"] = solubility_confidence(confidence_df)

    # Compute model metrics for RMSE
    rmse = np.sqrt(np.mean((confidence_df[target_column] - confidence_df["mean"]) ** 2))
    print(f"RMSE: {rmse} support: {len(confidence_df)}")

    # Domain Specific Confidence Threshold
    thres = 0.7

    # Make a 'high_confidence' column based on the threshold, casting to int (0, 1)
    confidence_df["high_confidence"] = (confidence_df["conf"] > thres).astype(int)

    # Now filter the data based on confidence and give RMSE for the filtered data
    high_confidence = confidence_df[confidence_df["high_confidence"] == 1]
    rmse = np.sqrt(np.mean((high_confidence[target_column] - high_confidence["mean"]) ** 2))
    print(f"RMSE: {rmse} support: {len(high_confidence)}")

    # Print out confusion matrix
    confusion_matrix(confidence_df)

    # Now for the high confidence subset
    confusion_matrix(high_confidence)

    # Columns of Interest
    dropdown_columns = ["interval", "conf", "mean_diff", "mean", "high_confidence", target_column]
    q_columns = [c for c in confidence_df.columns if c.startswith("q_")]
    dropdown_columns += q_columns
    dropdown_columns += feature_columns

    # Run the Unit Test on the Plugin
    plugin_test = PluginUnitTest(
        ScatterPlot,
        input_data=confidence_df[dropdown_columns],
        x="mean",
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
