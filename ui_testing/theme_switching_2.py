import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = html.Div(
    [
        dbc.Button("Switch to Dark Theme", id="theme-button", color="primary"),
        dcc.Graph(
            id="scatter-plot",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "scatter", "name": "SF"},
                ],
                "layout": {
                    "title": "Scatter Plot Example",
                    "plot_bgcolor": "#ffffff",
                    "paper_bgcolor": "#ffffff",
                    "font": {"color": "#000000"},
                },
            },
        ),
    ]
)


# Callback to switch themes dynamically
@app.callback(Output("scatter-plot", "figure"), [Input("theme-button", "n_clicks")], prevent_initial_call=True)
def switch_theme(n_clicks):
    dark_mode = n_clicks % 2 != 0
    if dark_mode:
        return {
            "data": [{"x": [1, 2, 3], "y": [4, 1, 2], "type": "scatter", "name": "SF"}],
            "layout": {
                "title": "Scatter Plot Example",
                "plot_bgcolor": "#333333",
                "paper_bgcolor": "#333333",
                "font": {"color": "#ffffff"},
            },
        }
    else:
        return {
            "data": [{"x": [1, 2, 3], "y": [4, 1, 2], "type": "scatter", "name": "SF"}],
            "layout": {
                "title": "Scatter Plot Example",
                "plot_bgcolor": "#ffffff",
                "paper_bgcolor": "#ffffff",
                "font": {"color": "#000000"},
            },
        }


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
