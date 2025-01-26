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
            # Main Content
            dbc.Row(
                [
                    # Column 1: Scatter Plot
                    dbc.Col(
                        [scatter_plot],
                        style={"flex": "1 1 auto", "min-width": "200px"},  # Shrinks but stays visible
                    ),
                    # Column 2: Molecular Viewer
                    dbc.Col(
                        [molecule_view],
                        style={"width": "480px", "flex": "0 0 auto"},  # Fixed width for right column
                        className="text-break",
                    ),
                ],
                style={"height": "90vh", "display": "flex", "flex-wrap": "nowrap"},  # Prevent wrapping
            ),
            # Update Button (hidden)
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
        style={"margin": "10px"},
    )
    return layout
