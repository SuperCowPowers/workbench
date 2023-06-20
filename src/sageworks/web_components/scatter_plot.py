"""A scatter plot component"""
from dash import dcc
import pandas as pd
import plotly.express as px
import numpy as np


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create(plot_id: str, df: pd.DataFrame) -> dcc.Graph:
    """Create a Scatter Plot"""

    # Fake data if it's not in the dataframe
    if "cluster" not in df.columns:
        df["cluster"] = np.random.randint(0, 10, df.shape[0])
        df["x"] = np.random.rand(df.shape[0])
        df["y"] = np.random.rand(df.shape[0])

    # Create the Scatter Plot
    color_map = px.colors.qualitative.Plotly
    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="y",
        color="cluster",
        log_x=True,
        size_max=16,
        title="Anomaly Cluster Plot",
        color_discrete_sequence=color_map,
    )
    return dcc.Graph(id="scatter_plot", figure=fig)
