"""A scatter plot component"""

import plotly.graph_objs
from dash import dcc
import pandas as pd
import plotly.express as px
import numpy as np


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(df: pd.DataFrame, title: str = "Compound Clusters") -> plotly.graph_objs.Figure:
    """Create a Scatter Plot
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        title (str): The title for the plot
    Returns:
        plotly.graph_objs.Figure: A Figure object containing the generated plots.
    """

    # Fake data if it's not in the dataframe
    if "cluster" not in df.columns:
        df["cluster"] = np.random.randint(0, 10, df.shape[0])
    if "x" not in df.columns:
        df["x"] = np.random.rand(df.shape[0])
    if "y" not in df.columns:
        df["y"] = np.random.rand(df.shape[0])

    # Since we're using discrete colors, we need to convert the cluster column to a string
    df["cluster"] = df["cluster"].astype(str)

    # Create the Scatter Plot
    color_map = px.colors.qualitative.Plotly
    fig = px.scatter(
        df,
        x="x",
        y="y",
        opacity=0.75,
        color="cluster",
        title=title,
        color_discrete_sequence=color_map,
    )
    fig.update_layout(title_y=0.97, title_x=0.3, title_xanchor="center", title_yanchor="top")
    fig.update_traces(
        marker=dict(size=14, line=dict(width=1, color="Black")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False)
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400)
    return fig


def create(component_id: str, df: pd.DataFrame, title: str = "Compound Clusters") -> dcc.Graph:
    """Create a Graph Component for scatter plots.

    Args:
        component_id (str): The ID of the UI component
        df (pd.DataFrame): A dataframe of data
        title (str): The title for the plot
    Returns:
        dcc.Graph: A Dash Graph Component representing the scatter plots.
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id=component_id, figure=create_figure(df, title=title))
