"""Proximity Utilities for Workbench"""

import logging
import pandas as pd

# Set up the log
log = logging.getLogger("workbench")


def target_intervals(pred_df, prox_df, id_column: str, target: str, distance_threshold=3.0) -> pd.DataFrame:
    """
    Add pred_min and pred_max columns to pred_df based on neighbors with
    distance < threshold in the proximity dataframe.

    Args:
        pred_df (DataFrame): Input dataframe with predictions and residuals
        prox_df (DataFrame): Proximity dataframe with distances
        id_column (str): Column name for the unique identifier (e.g., 'id')
        target (str): Column name for the target variable
        distance_threshold (float): Maximum distance for considering neighbors

    Returns:
        DataFrame: Original pred_df with added pred_min and pred_max columns
    """
    # Filter the proximity dataframe to include only rows with distance < threshold
    close_neighbors = prox_df[prox_df["distance"] < distance_threshold]

    # We only want the top 5 neighbors
    # close_neighbors = close_neighbors.groupby(id_column).head(5)

    # Group by id_column (gives you all the neighbors) and calculate min/max target values
    target_bounds = (
        close_neighbors.groupby(id_column)
        .agg(
            target_min=(target, "min"),
            target_max=(target, "max"),
            target_mean=(target, "mean"),
            target_std=(target, "std"),
        )
        .reset_index()
    )

    # Stddev can give NaN if all values are the same, so fill with 0
    target_bounds["target_std"] = target_bounds["target_std"].fillna(0)

    # Merge target bounds back to the original prediction dataframe
    result = pred_df.merge(target_bounds, on=id_column, how="left")

    # Calculate difference between targets
    result["target_delta"] = result["target_max"] - result["target_min"]

    # Prediction delta from target_mean
    result["pred_delta"] = (result["prediction"] - result["target_mean"]).abs()

    # Calculate the confidence (WIP)

    # 1. How close the prediction is to the target mean
    max_delta = 0.5
    result["confidence"] = 1 - (result["pred_delta"] / max_delta)

    # 2. How tight the spread is (normalized by some reasonable expected spread)
    max_expected_spread = 1.0  # Maximum expected range for normalization
    spread_conf = 1 - (result["target_std"] / max_expected_spread)

    # Combine the two confidence components with weighting
    spread_weight = 0.75  # Weight for the spread component
    result["confidence"] = (1 - spread_weight) * result["confidence"] + spread_weight * spread_conf

    # Clip confidence to ensure it's between 0 and 1
    result["confidence"] = result["confidence"].clip(lower=0, upper=1)

    return result


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import FeatureSet, Model, Endpoint, DFStore

    # Get predictions for the model
    recreate = False
    if recreate:
        model = Model("aqsol-regression-100")
        prox_model = Model("aqsol-prox")
        end = Endpoint(model.endpoints()[0])
        fs = FeatureSet(model.get_input())
        df = fs.pull_dataframe()
        pred_df = end.inference(df)

        # Save to the DFStore
        df_store = DFStore()
        df_store.upsert("/workbench/models/aqsol-regression-100/full_inference", pred_df)

        # Get the results of the proximity model
        prox_end = Endpoint(prox_model.endpoints()[0])
        prox_df = prox_end.inference(df)

        # Save to the DFStore
        df_store.upsert("/workbench/models/aqsol-prox/full_inference", prox_df)

    else:
        # Load the prediction and proximity dataframes from the DFStore
        df_store = DFStore()
        pred_df = df_store.get("/workbench/models/aqsol-regression-100/full_inference")
        prox_df = df_store.get("/workbench/models/aqsol-prox/full_inference")

    # Get the target intervals
    result_df = target_intervals(pred_df, prox_df, "id", "solubility")
    # result_df = prox_confidence(pred_df, prox_df, "id", "solubility")
    print("Prediction Intervals:")
    print(result_df)

    # Compute the prediction intervals based on the target intervals
    result_df["pred_interval"] = result_df["target_max"] - result_df["target_min"]

    # Make a plot of the prediction intervals
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    # show = ["id", "solubility", "prediction", "pred_min", "pred_max", "prediction_interval", "confidence"]
    show = [
        "id",
        "solubility",
        "prediction",
        "target_min",
        "target_max",
        "target_mean",
        "target_std",
        "pred_delta",
        "pred_interval",
        "confidence",
    ]

    plot = PluginUnitTest(ScatterPlot, input_data=result_df[show])
    plot.run()
    print("end")
