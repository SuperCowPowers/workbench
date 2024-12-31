"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def compound_explorer_layout(scatter_plot: dcc.Graph, molecule_view: html.Div) -> html.Div:
    """Set up the layout for the Compound Explorer Page"""
    layout = html.Div(
        children=[
            # Page Header
            dbc.Row([html.H2("Compound Explorer")]),
            # Scatter Plot
            dbc.Row([scatter_plot]),
            # Molecule Viewer
            dbc.Row([molecule_view]),
            # Scatter Plot and Molecule Viewer
            # dbc.Row(
            #     [
            #         # Column 1: Scatter Plot
            #         dbc.Col([scatter_plot], width=8),
            #         # Column 2: Molecular Viewer
            #         dbc.Col([molecule_view], width=4),
            #     ],
            # ),
            # Molecular Viewer for Neighbors
            # dbc.Row([molecule_view, molecule_view, molecule_view, molecule_view]),
            # Update Button
            html.Button("Update Plugin", id="update-button"),
            dcc.Tooltip(
                id="hover-tooltip", background_color="rgba(0,0,0,0)", border_color="rgba(0,0,0,0)", direction="top"
            ),
        ],
    )
    return layout
