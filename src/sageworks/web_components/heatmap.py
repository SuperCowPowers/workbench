"""A Heatmap component"""
import plotly.graph_objs
from dash import dcc
import pandas as pd
import plotly.graph_objects as go


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

    # Okay so the heatmap has inverse y-axis ordering so we need to flip the dataframe
    df = df.iloc[::-1]

    # Okay so there are numerous issues with getting back the index of the clicked on point
    # so we're going to store the indexes of the columns (this is SO stupid)
    x_labels = [f"{c}:{i}" for i, c in enumerate(df.columns)]
    y_labels = [f"{c}:{i}" for i, c in enumerate(df.index)]

    # Create the heatmap plot with custom settings
    height = max(400, len(df.index) * 50)
    fig = go.Figure(data=go.Heatmap(z=df, x=x_labels, y=y_labels, name="", colorscale=color_scale, zmin=-1, zmax=1))
    fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 0}, height=height)

    # Now remap the x and y axis labels (so they don't show the index)
    fig.update_xaxes(tickvals=x_labels, ticktext=df.columns, tickangle=30)
    fig.update_yaxes(tickvals=y_labels, ticktext=df.index)

    # Now we're going to customize the annotations and filter out low values
    label_threshold = 0.3
    for i, row in enumerate(df.index):
        for j, col in enumerate(df.columns):
            value = df.loc[row, col]
            if abs(value) > label_threshold:
                fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False)

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
