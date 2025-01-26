"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def compound_explorer_layout(scatter_plot: dcc.Graph, molecule_view: html.Div) -> html.Div:
    """Set up the layout for the Compound Explorer Page"""
    layout = html.Div(
        children=[
            # Page Header
            dbc.Row(
                [
                    html.H2(
                        [
                            "Compound Explorer: Public ",
                            html.A(
                                "(Sign up for Private Cloud)",
                                href="https://www.supercowpowers.com/workbench",
                                target="_blank",  # Opens the link in a new tab/window
                                style={
                                    "text-decoration": "none",  # Removes underline
                                    "font-size": "0.8em",  # Smaller font size
                                    "vertical-align": "middle",  # Aligns text properly with H2
                                },
                            ),
                        ]
                    ),
                ]
            ),
            # Scatter Plot
            dbc.Row([scatter_plot], style={"margin": "20px"}),
            # Molecule Viewer
            dbc.Row([molecule_view], style={"margin": "20px"}),
            # Molecular Viewer for Neighbors
            # dbc.Row([molecule_view, molecule_view, molecule_view, molecule_view]),
            # Update Button (Hidden)
            html.Button("Update Plugin", id="update-button", hidden=True),
            # Hover Tooltip
            dcc.Tooltip(
                id="hover-tooltip",
                background_color="rgba(0,0,0,0)",
                border_color="rgba(0,0,0,0)",
                direction="top",
                loading_text="",
            ),
        ],
    )
    return layout
