"""Layout for the Model Contests page"""

from dash import html, dcc

from workbench.utils.theme_manager import ThemeManager
from workbench.utils.config_manager import ConfigManager

# Get the UI update rate
update_rate = ConfigManager().ui_update_rate() * 1000  # Convert to Milliseconds

# Get the Project Name
tm = ThemeManager()
project_name = tm.branding().get("project_name", "Workbench")


def contests_layout() -> html.Div:
    layout = html.Div(
        children=[
            dcc.Interval(id="contests_refresh", interval=update_rate),
            # Server-side contest reports; the clientside renderer draws from this
            dcc.Store(id="contests_data"),
            dbc_header(),
            # The custom JS/CSS renderer owns everything inside this container.
            # Initial content is a spinner; the renderer replaces it once data arrives.
            html.Div(
                html.Div(
                    [html.Div(className="ct-spinner"), html.Div("Loading contests…")],
                    className="ct-spinner-wrap",
                ),
                id="contests-root",
                className="ct-root",
            ),
            # Dummy output target for the clientside render callback
            html.Div(id="contests_render_signal", style={"display": "none"}),
        ],
        style={"margin": "30px"},
    )
    return layout


def dbc_header() -> html.Div:
    return html.Div(
        html.H2(f"{project_name}: Model Contests"),
        style={"marginBottom": "18px"},
    )
