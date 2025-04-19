"""Layout for the Models page"""

from typing import Any
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
    model_plot: dcc.Graph,
    shap_plot: dcc.Graph,
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
                    dbc.Col(model_details, width=5, style={"padding": "30px 0px 0px 0px"}, className="text-break"),
                    # Column 2: Model Plot and Shap Summary
                    dbc.Col(
                        [
                            # Wrap this in a div to with className="workbench-container"
                            dbc.Row(
                                html.Div(model_plot, className="workbench-container"),
                                style={"padding": "20px 0px 0px 20px"}
                            ),
                            dbc.Row(shap_plot, style={"padding": "20px 0px 0px 20px"}),
                        ],
                        width=7,
                    ),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
