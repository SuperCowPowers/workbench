"""A confusion matrix component"""
from dash import dcc
import plotly.figure_factory as ff


def create_figure(matrix: list[list[float]]):
    """Create the Figure for the Confusion Matrix"""

    x = ["low", "medium", "high"]
    y = ["high", "medium", "low"]
    z = matrix

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, zmin=0, zmax=1)  # colorscale='Viridis')

    # add title
    fig.update_layout(
        title_text="<i><b>Confusion matrix</b></i>",
        # xaxis = dict(title='x'),
        # yaxis = dict(title='x')
    )

    # add custom x-axis title
    fig.add_annotation(
        dict(
            font=dict(size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom y-axis title
    fig.add_annotation(
        dict(
            font=dict(size=14),
            x=-0.20,
            y=0.5,
            showarrow=False,
            text="Actual value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=80))

    # add colorbar
    fig["data"][0]["showscale"] = True
    return fig


def create(matrix: list[list[float]]) -> dcc.Graph:
    """Create a Confusion Matrix"""

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Graph(id="confusion_matrix", figure=create_figure(matrix))
