"""DataSources Layout: Layout for the DataSources page in the Artifact Viewer"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def data_sources_layout(components: dict) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: DataSources (Mockup)"),
                    html.Div(
                        "Last Updated: ",
                        id="last-updated-data-sources",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 160px",
                        },
                    ),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            dbc.Row(components["data_sources_details"]),
            dbc.Row(components["scatter_plot"]),
            dcc.Interval(id="data-sources-updater", interval=5000, n_intervals=0),
        ],
        style={"margin": "30px"},
    )
    return layout
