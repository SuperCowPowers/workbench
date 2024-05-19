import dash
from dash import dcc, html, Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    [
        dcc.Dropdown(
            id="theme-selector",
            options=[
                {"label": "Light Theme", "value": "scatter-plot-light"},
                {"label": "Dark Theme", "value": "scatter-plot-dark"},
            ],
            value="scatter-plot-light",  # Default value
        ),
        dcc.Graph(
            id="scatter-plot",
            className="dash-graph scatter-plot-light",  # Initial class name and theme
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


# Callback to update the figure based on theme selection
@app.callback(Output("scatter-plot", "figure"), [Input("theme-selector", "value")])
def update_scatter_plot_figure(selected_theme):
    if selected_theme == "scatter-plot-dark":
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
