"""A Violin Plot component"""
import plotly.graph_objs
from dash import dcc
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from enum import Enum
import math


class PlotType(Enum):
    """Enumeration for the plot type"""

    violin = go.Violin
    box = go.Box


def compute_subplot_layout(n):
    """
    Compute a 'nice' layout for a given number of subplots.
    Layout: maximum of 8 columns aiming for a rectangular grid and adding rows as needed
    Args:
        n (int): The total number of subplots.
    Returns:
        tuple: A tuple (rows, cols) representing the layout.
    """

    # Start with a single row
    rows = 1
    while True:
        cols = math.ceil(n / rows)
        if cols <= 8:
            return int(rows), int(cols)
        else:
            rows += 1


def calculate_height(num_rows: int):
    # Set the base height
    base_height = 300
    if num_rows == 1:
        return base_height
    return base_height + num_rows * 80


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(df: pd.DataFrame, plot_type: str, figure_args: dict, max_plots: int) -> plotly.graph_objs.Figure:
    """Create a set of plots for the numeric columns in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        plot_type (str): The type of plot to create. Valid values are 'violin' and 'box'.
        figure_args (dict): A dictionary of arguments to pass to the plot object.
            For violin plot arguments, refer to: https://plotly.com/python/reference/violin/
            For box plot arguments, refer to: https://plotly.com/python/reference/box/
        max_plots (int): The maximum number of plots to create.

    Returns:
        plotly.graph_objs.Figure: A Figure object containing the generated plots.
    """

    if plot_type not in list(PlotType.__members__.keys()):
        raise ValueError("Invalid plot type")

    figure_object = PlotType[plot_type].value

    # Sanity check the dataframe
    if df is None or df.empty or list(df.columns) == ["uuid", "status"]:
        return go.Figure()

    numeric_columns = list(df.select_dtypes("number").columns)
    numeric_columns = [col for col in numeric_columns if df[col].nunique() > 1]  # Only columns > 1 unique value

    # HARDCODE: Not sure how to get around hard coding these columns
    not_show = [
        "id",
        "Id",
        "ID",
        "uuid",
        "write_time",
        "api_invocation_time",
        "is_deleted",
        "x",
        "y",
        "cluster",
        "outlier_group",
    ]
    numeric_columns = [col for col in numeric_columns if col not in not_show]
    numeric_columns = numeric_columns[:max_plots]  # Max plots

    # Compute the number of rows and columns
    num_plots = len(numeric_columns)
    num_rows, num_columns = compute_subplot_layout(num_plots)
    fig = make_subplots(rows=num_rows, cols=num_columns, vertical_spacing=0.07)
    for i, col in enumerate(numeric_columns):
        fig.add_trace(
            figure_object(y=df[col], name=col, **figure_args),
            row=i // num_columns + 1,
            col=i % num_columns + 1,
        )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=(calculate_height(num_rows)),
        dragmode="select",
        newselection=dict(line=dict(color="grey", width=1, dash="dot")),
    )
    fig.update_traces(selected_marker=dict(size=10, color="white"), selector=dict(type="violin"))
    # fig.update_traces(selected_marker_color="white", selector=dict(type="violin"))
    fig.update_traces(unselected_marker=dict(size=6, opacity=0.5), selector=dict(type="violin"))
    fig.update_traces(
        box_line_color="rgba(255, 255, 255, 0.75)",
        meanline_color="rgba(255, 255, 255, 0.75)",
        width=0.5,
        selector=dict(type="violin"),
    )
    return fig


def create(
    component_id: str,
    df: pd.DataFrame,
    plot_type: str,
    figure_args: dict,
    max_plots: int,
) -> dcc.Graph:
    """Create a Graph Component for vertical distribution plots.

    Args:
        component_id (str): The ID of the UI component.
        df (pd.DataFrame): A dataframe of data.
        plot_type (str): The type of plot to create. Valid values are 'violin' and 'box'.
        figure_args (dict): A dictionary of arguments to pass to the plot object.
            For violin plot arguments, refer to: https://plotly.com/python/reference/violin/
            For box plot arguments, refer to: https://plotly.com/python/reference/box/
        max_plots (int): The maximum number of plots to create.

    Returns:
        dcc.Graph: A Dash Graph Component representing the vertical distribution plots.
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id=component_id, figure=create_figure(df, plot_type, figure_args, max_plots))
