"""A box plot component"""
from dash import dcc
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Create a set of Box Plots for the numeric columns in the dataframe"""

    numeric_columns = list(df.select_dtypes('number').columns)[:10]  # Max 10 columns
    numeric_columns = [col for col in numeric_columns if len(df[col].unique()) > 1]  # Only columns > 1 unique value
    numeric_columns = [col for col in numeric_columns if col not in ['id', 'Id', 'ID', 'Id_']]  # Remove id columns
    fig = make_subplots(rows=1, cols=len(numeric_columns))
    for i, col in enumerate(numeric_columns):
        fig.add_trace(
            go.Box(y=df[col], name=col, notched=True, showlegend=False),
            row=1, col=i+1
        )
    fig.update_traces(boxpoints='all', jitter=.2)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return dcc.Graph(id="box_plot", figure=fig)


def create(df: pd.DataFrame) -> dcc.Graph:
    """Create a Violin Plot Graph Component"""

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id="box_plot", figure=create_figure(df))
