"""A Violin Plot component"""
import plotly.graph_objs
from dash import dcc
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def compute_rows_columns(num_plots):
    """Errr... I think this works but happy to be improved"""
    max_columns = 8
    overflow = 1 if num_plots % max_columns != 0 else 0
    num_rows = num_plots//max_columns + overflow
    num_columns = round(num_plots/num_rows + .1)
    return num_rows, num_columns


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(df: pd.DataFrame) -> plotly.graph_objs.Figure:
    """Create a set of Violin Plots for the numeric columns in the dataframe"""

    numeric_columns = list(df.select_dtypes('number').columns)
    numeric_columns = [col for col in numeric_columns if len(df[col].unique()) > 1]  # Only columns > 1 unique value
    numeric_columns = [col for col in numeric_columns if col not in ['id', 'Id', 'ID', 'Id_']]  # Remove id columns
    numeric_columns = numeric_columns[:24]  # Max 24 plots

    # Compute the number of rows and columns
    num_plots = len(numeric_columns)
    num_rows, num_columns = compute_rows_columns(num_plots)
    print(f"Creating {num_plots} violin plots in {num_rows} rows and {num_columns} columns")
    fig = make_subplots(rows=num_rows, cols=num_columns)
    for i, var in enumerate(numeric_columns):
        fig.add_trace(
            go.Violin(y=df[var], name=var, box_visible=True, meanline_visible=True, showlegend=False, points="all"),
            row=i//num_columns+1, col=i % num_columns+1
        )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


def create(df: pd.DataFrame) -> dcc.Graph:
    """Create a Violin Plot Graph Component"""

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id="violin_plot", figure=create_figure(df))
