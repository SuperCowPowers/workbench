"""Layout for the model scoreboard"""
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


def artifact_layout(app: Dash, components: dict) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(children=[
        dbc.Row([html.H2(app.title), html.Div('Last Updated: ', id='last-updated',
                                              style={'color': 'rgb(140, 200, 140)',
                                                     'fontSize': 15, 'padding': '0px 0px 0px 120px'})]),
        dbc.Row(html.H3('Incoming Data'), style={'padding': '20px 0px 0px 0px'}),
        dbc.Row(components['incoming_data']),
        dbc.Row(html.H3('Data Sources'), style={'padding': '20px 0px 0px 0px'}),
        dbc.Row(components['data_sources']),
        dbc.Row(html.H3('Feature Sets'), style={'padding': '20px 0px 0px 0px'}),
        dbc.Row(components['feature_sets']),
        dbc.Row(html.H3('Models'), style={'padding': '20px 0px 0px 0px'}),
        dbc.Row(components['models']),
        dbc.Row(html.H3('Endpoints'), style={'padding': '20px 0px 0px 0px'}),
        dbc.Row(components['endpoints']),
        dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
        ], style={'margin': '30px'})
    return layout
