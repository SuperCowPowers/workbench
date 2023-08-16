"""A Heatmap component"""
import plotly.graph_objs
from dash import dcc
import pandas as pd
import plotly.express as px


# For heatmaps see (https://plotly.com/python/heatmaps/)
def create_figure(df: pd.DataFrame) -> plotly.graph_objs.Figure:
    """Create a heatmap plot for the numeric columns in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data for the heatmap.
    Returns:
        plotly.graph_objs.Figure: A Figure object containing the heatmap.
    """

    # A nice color scale for the heatmap
    color_scale = [
        [0, "rgb(64,64,128)"],
        [0.15, "rgb(48, 120, 120)"],
        [0.4, "rgb(32, 32, 32)"],
        [0.5, "rgb(32, 32, 32)"],
        [0.6, "rgb(32, 32, 32)"],
        [0.85, "rgb(120, 120, 48)"],
        [1.0, "rgb(128, 64, 64)"],
    ]

    # Create the imshow plot with custom settings
    fig = px.imshow(df, color_continuous_scale=color_scale, range_color=[-1, 1], text_auto=".2f")
    fig.update_layout(
        margin={"t": 30, "b": 0, "r": 0, "l": 0, "pad": 0},
        autosize=True,
    )
    return fig


def create(component_id: str, df: pd.DataFrame) -> dcc.Graph:
    """Create a Graph Component for a heatmap plot.

    Args:
        component_id (str): The ID of the UI component.
        df (pd.DataFrame): A dataframe in the format given by df.corr()

    Returns:
        dcc.Graph: A Dash Graph Component representing the vertical distribution plots.
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id=component_id, figure=create_figure(df))
