"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.config_manager import ConfigManager

# Get the UI update rate
update_rate = ConfigManager().ui_update_rate() * 1000  # Convert to Milliseconds

# Get the Project Name
tm = ThemeManager()
project_name = tm.branding().get("project_name", "Workbench")


def feature_sets_layout(
    feature_sets_table: AGTable,
    feature_set_sample_rows: AGTable,
    feature_set_details: dcc.Markdown,
    violin_plot: dcc.Graph,
    correlation_matrix: dcc.Graph,
) -> html.Div:
    # The layout for the FeatureSets page
    layout = html.Div(
        children=[
            dcc.Interval(id="feature_sets_refresh", interval=update_rate),
            dcc.Store(id="feature_sets_page_loaded", data=False),
            dbc.Row(
                [
                    html.H2(f"{project_name}: FeatureSets"),
                ]
            ),
            # A table that lists out all the Feature Sets
            dbc.Row(feature_sets_table),
            # Sample Rows for the selected Feature Set
            dbc.Row(
                html.H3("Sampled Rows", id="feature_sample_rows_header", style={"margin": "30px 0px 10px 0px"}),
            ),
            dbc.Row(
                feature_set_sample_rows,
            ),
            # Row: Data Source Details, Column2: Violin Plots, Correlation Matrix
            dbc.Row(
                [
                    # Column 1: Feature Set Details
                    dbc.Col(
                        feature_set_details,
                        width=4,
                        className="text-break workbench-container",
                        style={"margin": "20px 0px 0px 0px", "padding": "20px"},
                    ),
                    # Column 2: Violin Plots and Correlation Matrix
                    dbc.Col(
                        [
                            dbc.Row(
                                violin_plot, className="workbench-container", style={"margin": "20px 0px 10px 20px"}
                            ),
                            dbc.Row(
                                correlation_matrix,
                                className="workbench-container",
                                style={"margin": "20px 0px 10px 20px"},
                            ),
                        ],
                        width=8,
                        style={"padding": "0px"},
                    ),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
