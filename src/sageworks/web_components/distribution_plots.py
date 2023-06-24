"""A Violin Plot component"""
import plotly.graph_objs
from dash import dcc
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from enum import Enum


class PlotType(Enum):
    """The type of plot"""

    violin = go.Violin
    box = go.Box


def compute_rows_columns(num_plots):
    """Errr... I think this works but happy to be improved"""
    max_columns = 8
    overflow = 1 if num_plots % max_columns != 0 else 0
    num_rows = num_plots // max_columns + overflow
    num_columns = round(num_plots / num_rows + 0.1)
    return num_rows, num_columns


def calculate_height(num_rows: int):
    # Set the base height
    base_height = 300
    if num_rows == 1:
        return base_height
    return base_height + num_rows * 170


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(
    df: pd.DataFrame, plot_type: str, figure_args: dict, max_plots: int
) -> plotly.graph_objs.Figure:
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
    numeric_columns = [
        col for col in numeric_columns if len(df[col].unique()) > 1
    ]  # Only columns > 1 unique value
    numeric_columns = [
        col for col in numeric_columns if col not in ["id", "Id", "ID", "Id_"]
    ]  # Remove id columns
    numeric_columns = numeric_columns[:max_plots]  # Max plots

    # Compute the number of rows and columns
    num_plots = len(numeric_columns)
    num_rows, num_columns = compute_rows_columns(num_plots)
    # print(f"Creating {num_plots} violin plots in {num_rows} rows and {num_columns} columns")
    fig = make_subplots(rows=num_rows, cols=num_columns)
    for i, col in enumerate(numeric_columns):
        fig.add_trace(
            figure_object(y=df[col], name=col, **figure_args),
            row=i // num_columns + 1,
            col=i % num_columns + 1,
        )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(height=(calculate_height(num_rows)))
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
    return dcc.Graph(
        id=component_id, figure=create_figure(df, plot_type, figure_args, max_plots)
    )
