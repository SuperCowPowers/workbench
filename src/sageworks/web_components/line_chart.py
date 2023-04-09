"""A line chart component"""
from dash import dcc
import pandas as pd
import plotly.express as px


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create(df: pd.DataFrame = None) -> dcc.Graph:
    """Create a Line Chart"""

    df = px.data.stocks(indexed=True) - 1
    df.columns.name = "Endpoints"
    # TEMP
    df.rename(
        {
            "Date": "awesome",
            "GOOG": "aqsol-regression 1",
            "AAPL": "aqsol-regression 2",
            "AMZN": "abalone-regression 1",
            "FB": "abalone-regression 2",
            "NFLX": "super-secret 1",
            "MSFT": "super-secret 2",
        },
        axis=1,
        inplace=True,
    )

    fig = px.area(df, facet_col="Endpoints", facet_col_wrap=2)
    return dcc.Graph(id="line_chart", figure=fig)
