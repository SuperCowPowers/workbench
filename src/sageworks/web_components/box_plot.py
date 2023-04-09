"""A box plot component"""
from dash import dcc
import pandas as pd
import plotly.express as px


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create(df: pd.DataFrame = None) -> dcc.Graph:
    """Create a Box Plot"""

    df = px.data.tips()

    # TEMP
    df.rename(
        {"total_bill": "awesome", "time": "Variable Comparison"},
        axis=1,
        inplace=True,
    )
    df.replace(
        {
            "Dinner": "Good Feature",
            "Lunch": "Better Feature",
        },
        inplace=True,
    )

    fig = px.box(df, x="Variable Comparison", y="awesome", color="Variable Comparison", points="all")
    return dcc.Graph(id="line_chart", figure=fig)
