"""An Outlier Plot Component"""

import plotly.graph_objs
from dash import dcc
import pandas as pd
import numpy as np
import plotly.express as px
import logging

# SageWorks Imports
from sageworks.algorithms.dataframe.aggregation import aggregate
from sageworks.algorithms.dataframe.dimensionality_reduction import DimensionalityReduction


# SageWorks Logger
log = logging.getLogger("sageworks")


# For colormaps see (https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express)
def create_figure(df: pd.DataFrame, title: str = "Outlier Groups") -> plotly.graph_objs.Figure:
    """Create a Outlier Plot
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        title (str): The title for the plot
    Returns:
        plotly.graph_objs.Figure: A Figure object containing the generated plots.
    """

    # The dataframe is required to have the following columns:
    #   outlier_group: The outlier group descriptor (e.g. "mol_log_p_high")
    if "outlier_group" not in df.columns:
        raise ValueError("Outlier Plot requires column 'outlier_group' to be in the dataframe.")

    # Now process the dataframe
    # - aggregate the dataframe by outlier_group
    # - compute the mean of each group
    # - run dimensionality reduction on the mean values
    if "x" not in df.columns:
        log.info("Outlier Plot: No coordinates in dataframe, running aggregation and dimensionality reduction...")
        agg_df = aggregate(df, group_column="outlier_group")
        coord_df = DimensionalityReduction().fit_transform(agg_df)
    else:
        log.info("Outlier Plot: Coordinates found...")
        coord_df = df

    # Scale the group_count column so that it's between 5 and 100
    min_size = 5
    max_size = 100
    coord_df["group_count"] = np.interp(
        coord_df["group_count"],
        (coord_df["group_count"].min(), coord_df["group_count"].max()),
        (min_size, max_size),
    )

    # Sample column needs to be first
    coord_df["is_sample"] = coord_df["outlier_group"] == "sample"
    coord_df.sort_values("is_sample", ascending=False, inplace=True)
    coord_df.drop("is_sample", axis=1, inplace=True)

    # Create the Outlier Plot
    color_map = px.colors.qualitative.Plotly
    fig = px.scatter(
        coord_df,
        x="x",
        y="y",
        opacity=0.75,
        color="outlier_group",
        size="group_count",
        size_max=100,
        title=title,
        color_discrete_sequence=color_map,
    )

    # Make sure the 'sample' group is always grey
    for trace in fig.data:
        if "sample" in trace.name:
            trace.marker.color = "darkgrey"

    # Various Plotly Layout Updates
    fig.update_layout(title_y=0.97, title_x=0.3, title_xanchor="center", title_yanchor="top")
    fig.update_traces(
        marker=dict(line=dict(width=2, color="black")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False)
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400)
    return fig


def create(component_id: str, df: pd.DataFrame, title: str = "Outlier Groups") -> dcc.Graph:
    """Create a Graph Component for outlier plots.

    Args:
        component_id (str): The ID of the UI component
        df (pd.DataFrame): A dataframe of data
        title (str): The title for the plot
    Returns:
        dcc.Graph: A Dash Graph Component representing the vertical distribution plots.
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id=component_id, figure=create_figure(df, title=title))
