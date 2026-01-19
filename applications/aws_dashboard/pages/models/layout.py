"""Layout for the Models page"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.config_manager import ConfigManager

# Get the UI update rate
update_rate = ConfigManager().ui_update_rate() * 1000  # Convert to Milliseconds

# Get the Project Name
tm = ThemeManager()
project_name = tm.branding().get("project_name", "Workbench")


def models_layout(
    models_table: AGTable,
    model_details: html.Div,
    model_plot: html.Div,
    shap_plot: html.Div,
) -> html.Div:
    layout = html.Div(
        children=[
            dcc.Interval(id="models_refresh", interval=update_rate),
            dcc.Store(id="models_page_loaded", data=False),
            dbc.Row(
                [
                    html.H2(f"{project_name}: Models"),
                ]
            ),
            # A table that lists out all the Models
            dbc.Row(models_table),
            # Model Details, Model Plot, and Shap Summary Plot
            dbc.Row(
                [
                    # Column 1: Model Details
                    dbc.Col(
                        dcc.Loading(model_details, type="dot", color="#33aa33", delay_show=1000),
                        width=4,
                        className="text-break workbench-container",
                        style={"margin": "20px 0px 0px 0px", "padding": "20px"},
                    ),
                    # Column 2: Model Plot and Shap Summary
                    dbc.Col(
                        [
                            dbc.Row(
                                dcc.Loading(
                                    model_plot,
                                    type="graph",
                                    delay_show=500,
                                ),
                                className="workbench-container",
                                style={"margin": "20px 0px 10px 20px"},
                            ),
                            dbc.Row(
                                dcc.Loading(shap_plot, type="graph", delay_show=500),
                                className="workbench-container",
                                style={"margin": "20px 0px 10px 20px"},
                            ),
                        ],
                        width=8,
                        style={"padding": "0px"},
                    ),
                ],
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
