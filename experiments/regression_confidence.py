from typing import Union
import pandas as pd
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.web_components.plugins.scatter_plot import ScatterPlot
from sageworks.web_components.plugin_unit_test import PluginUnitTest

# Run an integration test
from pprint import pprint
import numpy as np
from sageworks.api import Model, Endpoint

# Get the endpoint
end = Endpoint("aqsol-qr-end")

# Domain specific error_distance (if we got it wrong by this much, we have low confidence)
error_distance = 1.5

# Get the endpoint model and target column
model = Model(end.get_input())
target = model.target()

# Grab the inference data
pred_df = model.get_inference_predictions()  # "qrr_2024_07_11")

# Domain specific confidence
pred_df["target_spread"] = pred_df["q_95"] - pred_df["q_05"]
pred_df["target_confidence"] = np.clip(1 - (pred_df["target_spread"] / (error_distance * 4.0)), 0, 1)

# Compute the regression outliers
regression_outliers = np.maximum(np.abs(pred_df["qr_05"]), np.abs(pred_df["qr_95"]))

# Compute regression IQR distance
pred_df["iqr"] = pred_df["qr_75"] - pred_df["qr_25"]

# Compute residual confidence
pred_df["residual_confidence"] = np.clip(1 - (regression_outliers / error_distance), 0, 1)
# pred_df["residual_confidence"] = np.clip(1 - (pred_df["iqr"] / 2.0), 0, 1)

# Compute the median delta for the prediction
pred_df["median_delta"] = np.abs(pred_df["q_50"] - pred_df["prediction"])
pred_df["median_confidence"] = np.clip(1 - (pred_df["median_delta"] / (error_distance / 2)), 0, 1)

# Confidence is the product of target, residual, and median confidence
# pred_df["confidence"] = pred_df["residual_confidence"] * pred_df["median_confidence"]
# pred_df["confidence"] = pred_df["residual_confidence"] * pred_df["target_confidence"]
pred_df["confidence"] = pred_df["residual_confidence"]

# Grab the performance metrics
metrics = model.get_inference_metrics()
pprint(metrics)

# Confidence Threshold
confidence_thres = 0.5

# Now filter the data based on confidence and give RMSE for the filtered data
predictions_filtered = pred_df[pred_df["confidence"] > confidence_thres]
rmse_filtered = np.sqrt(np.mean((predictions_filtered[target] - predictions_filtered["prediction"]) ** 2))
print(f"RMSE Filtered: {rmse_filtered} support: {len(predictions_filtered)}")

# Columns that we want to show when we hover above a point
hover_columns = [
    "q_05",
    "q_25",
    "q_50",
    "q_75",
    "q_95",
    "qr_05",
    "qr_25",
    "qr_50",
    "qr_75",
    "qr_95",
    "prediction",
    target,
    "confidence",
    "target_spread",
    "target_confidence",
    "residual_confidence",
    "median_delta",
    "median_confidence",
]

PluginUnitTest(
    ScatterPlot, input_data=pred_df, x="solubility", y="prediction", color="confidence", hover_columns=hover_columns
).run()
