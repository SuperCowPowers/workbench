"""A histogram component"""
from dash import dcc
import pandas as pd
import plotly.express as px


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create(df: pd.DataFrame = None, variant=1) -> dcc.Graph:
    """Create a Histogram Plot"""
    gap_df = px.data.gapminder()

    # TEMP
    gap_df.rename(
        {
            "gdpPercap": "awesome",
            "lifeExp": "stuff",
            "continent": "Project",
            "pop": "logS",
        },
        axis=1,
        inplace=True,
    )
    gap_df.replace(
        {
            "Asia": "Project X",
            "Europe": "Project Y",
            "Africa": "Project Z",
            "Americas": "Project A",
            "Oceania": "Project B",
        },
        inplace=True,
    )
    if variant == 1:
        gap_df = gap_df[gap_df["awesome"] < 10000]
    else:
        gap_df = gap_df[gap_df["awesome"] > 20000]

    fig = px.histogram(
        gap_df,
        x="awesome",
        y="stuff",
        color="Project",
        title="Cool Stuff",
        color_discrete_sequence=px.colors.qualitative.Antique,
    )
    return dcc.Graph(id="histogram", figure=fig)
