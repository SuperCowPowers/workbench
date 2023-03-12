"""Layout for the model scoreboard"""
from dash import Dash, html
import dash_bootstrap_components as dbc


def artifact_layout(app: Dash, components: dict) -> html.Div:
    layout = html.Div(children=[
        dbc.Row(html.H2(app.title)),
        # Just put all the tables in as Rows for Now (do something fancy later)
        dbc.Row(html.H3('Incoming Data'), style={'padding': '50px 0px 0px 0px'}),
        dbc.Row(components['incoming_data']),
        dbc.Row(html.H3('Data Sources'), style={'padding': '50px 0px 0px 0px'}),
        dbc.Row(components['data_sources']),
        dbc.Row(html.H3('Feature Sets'), style={'padding': '50px 0px 0px 0px'}),
        dbc.Row(components['feature_sets']),
        dbc.Row(html.H3('Models'), style={'padding': '50px 0px 0px 0px'}),
        dbc.Row(components['models']),
        dbc.Row(html.H3('Endpoints'), style={'padding': '50px 0px 0px 0px'}),
        dbc.Row(components['endpoints']),
        ], style={'margin': '30px'})
    return layout
