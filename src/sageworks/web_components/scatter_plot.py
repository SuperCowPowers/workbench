"""A scatter plot component"""
from dash import dcc
import pandas as pd
import plotly.express as px


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create(plot_id: str, df: pd.DataFrame) -> dcc.Graph:
    """Create a Scatter Plot"""

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
