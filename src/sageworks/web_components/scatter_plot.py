"""A scatter plot component"""
import plotly.graph_objs
from dash import dcc
import pandas as pd
import plotly.express as px
import numpy as np


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(df: pd.DataFrame) -> plotly.graph_objs.Figure:
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
    return fig


def create(component_id: str, df: pd.DataFrame) -> dcc.Graph:
    """Create a Graph Component for vertical distribution plots.

    Args:
        component_id (str): The ID of the UI component.
        df (pd.DataFrame): A dataframe of data.
    Returns:
        dcc.Graph: A Dash Graph Component representing the vertical distribution plots.
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id=component_id, figure=create_figure(df))
