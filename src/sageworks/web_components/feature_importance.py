"""Feature Importance component"""
from dash import dcc
import pandas as pd
import plotly.express as px


def create_figure(feature_importance: dict, orientation="h") -> dcc.Graph:
    """Create the Figure for a Feature Importance Chart"""
    df = pd.DataFrame(
        {
            "Feature": feature_importance.keys(),
            "Importance": feature_importance.values(),
        },
    )
    df = df.sort_values(by="Importance")
    fig = px.bar(df, x="Importance", y="Feature", orientation=orientation)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def create(feature_importance: dict, orientation="h") -> dcc.Graph:
    """Create a Feature Importance Chart"""
    return dcc.Graph(id="feature_importance", figure=create_figure(feature_importance, orientation))
