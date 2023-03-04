"""Layout for the model scoreboard"""
from dash import Dash, html
import dash_bootstrap_components as dbc


def scoreboard_layout(app: Dash, components: dict) -> html.Div:
    layout = html.Div(children=[
            dbc.Row(html.H2(app.title)),
            dbc.Row([
                # Model Table and Model Details
                dbc.Col([
                        dbc.Row(components['model_table']),
                        dbc.Row(html.H3('Model Details'), style={'padding': '50px 0px 0px 0px'}),
                        dbc.Row([dbc.Col(components['model_details']), dbc.Col(components['confusion_matrix'])]),
                        dbc.Row(components['scatter_plot'])
                        ], width=8),

                # Feature Importance and Details
                dbc.Col([
                    dbc.Row(html.H3('Feature Importance')),
                    dbc.Row(components['feature_importance']),
                    dbc.Row(html.H3('Feature Details', style={'padding': '10px 0px 0px 0px'})),
                    dbc.Row(components['feature_details'])
                ], width=4)
            ]),

        ], style={'margin': '30px'})
    return layout


def scoreboard_layout_alt1(app: Dash, components: dict) -> html.Div:
    layout = html.Div(children=[
        dbc.Row(html.H2(app.title)),
        dbc.Row([
            dbc.Col(components['model_table'], width=8),
            dbc.Col(components['confusion_matrix'], width=4)]
        ),
        dbc.Row(components['scatter_plot'])
    ])
    return layout


def scoreboard_layout_alt2(app: Dash, components: dict) -> html.Div:
    layout = html.Div(children=[
        dbc.Row(html.H2(app.title)),
        dbc.Row(html.H2('  ')),
        dbc.Row([
            dbc.Col(components['model_table'], width=7),
            dbc.Col(components['feature_importance'], width=5)]
        ),
        dbc.Row(html.H2('Model Details')),
        dbc.Row([
            dbc.Col(components['model_details'], width=4),
            dbc.Col(components['confusion_matrix'], width=4),
            dbc.Col(components['scatter_plot'], width=4)]
        )
    ])
    return layout
