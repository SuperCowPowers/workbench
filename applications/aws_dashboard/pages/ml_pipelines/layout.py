"""Layout for the ML Pipelines page"""

from dash import html, dcc

from workbench.utils.theme_manager import ThemeManager
from workbench.utils.config_manager import ConfigManager

# Get the UI update rate
update_rate = ConfigManager().ui_update_rate() * 1000  # Convert to Milliseconds

# Get the Project Name
tm = ThemeManager()
project_name = tm.branding().get("project_name", "Workbench")


def ml_pipelines_layout() -> html.Div:
    layout = html.Div(
        children=[
            dcc.Interval(id="ml_pipelines_refresh", interval=update_rate),
            # Server-side pipeline hierarchy; the clientside renderer draws from this
            dcc.Store(id="ml_pipelines_data"),
            dbc_header(),
            # The custom JS/CSS renderer owns everything inside this container.
            # Initial content is a spinner; the renderer replaces it once data arrives.
            html.Div(
                html.Div(
                    [html.Div(className="mlp-spinner"), html.Div("Loading pipelines…")],
                    className="mlp-spinner-wrap",
                ),
                id="ml-pipelines-root",
            ),
            # Dummy output target for the clientside render callback
            html.Div(id="ml_pipelines_render_signal", style={"display": "none"}),
        ],
        style={"margin": "30px"},
    )
    return layout


def dbc_header() -> html.Div:
    return html.Div(
        html.H2(f"{project_name}: ML Pipelines"),
        style={"marginBottom": "18px"},
    )
