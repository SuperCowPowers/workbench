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
        [0.35, "rgb(40, 40, 40)"],
        [0.5, "rgb(40, 40, 40)"],
        [0.65, "rgb(40, 40, 40)"],
        [0.85, "rgb(120, 120, 48)"],
        [1.0, "rgb(128, 64, 64)"],
    ]

    # Create the imshow plot with custom settings
    height = max(400, len(df.index) * 50)
    fig = px.imshow(df, color_continuous_scale=color_scale, range_color=[-1, 1])
    fig.update_layout(
        margin={"t": 30, "b": 10, "r": 10, "l": 10, "pad": 0},
        height=height
    )
    fig.update_xaxes(tickangle=30)

    # Now we're going to customize the annotations and filter out low values
    label_threshold = 0.3
    for i, row in enumerate(df.index):
        for j, col in enumerate(df.columns):
            value = df.loc[row, col]
            if abs(value) > label_threshold:
                fig.add_annotation(x=j, y=i, text=f'{value:.2f}', showarrow=False)

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
