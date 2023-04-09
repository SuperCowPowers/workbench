"""A scatter plot component"""
from dash import dcc
import pandas as pd
import plotly.express as px


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create(df: pd.DataFrame = None, variant=1) -> dcc.Graph:
    """Create a Scatter Plot"""
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

    color_map = px.colors.qualitative.Plotly
    if variant == 1:
        pass
    elif variant == 2:
        gap_df["awesome"] = 50000 - gap_df["awesome"]
    else:
        gap_df.replace(
            {
                "Project X": "Training",
                "Project Y": "Training",
                "Project Z": "Training",
                "Project A": "Testing",
                "Project B": "Testing",
            },
            inplace=True,
        )
    fig = px.scatter(
        gap_df,
        x="awesome",
        y="stuff",
        size="logS",
        color="Project",
        log_x=True,
        size_max=60,
        title="Cool Stuff",
        color_discrete_sequence=color_map,
    )
    return dcc.Graph(id="scatter_plot", figure=fig)
