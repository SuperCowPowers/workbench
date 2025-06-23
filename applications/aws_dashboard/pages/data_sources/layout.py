"""DataSources Layout: Layout for the DataSources page in the Artifact Viewer"""

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


def data_sources_layout(
    data_sources_table: AGTable,
    data_source_sample_rows: AGTable,
    data_source_details: dcc.Markdown,
    violin_plot: dcc.Graph,
    correlation_matrix: dcc.Graph,
) -> html.Div:
    # The layout for the DataSources page
    layout = html.Div(
        children=[
            dcc.Interval(id="data_sources_refresh", interval=update_rate),
            dcc.Store(id="data_sources_page_loaded", data=False),
            dbc.Row(
                [
                    html.H2(f"{project_name}: DataSources"),
                ]
            ),
            # A table that lists out all the Data Sources
            dbc.Row(data_sources_table),
            # Sample/Outlier Rows for the selected Data Source AND Outlier Plot
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Sampled Rows", id="sample_rows_header", style={"margin": "30px 0px 10px 0px"}),
                            ),
                            dbc.Row(
                                data_source_sample_rows,
                            ),
                        ],
                    ),
                ]
            ),
            # Column1: Data Source Details, Column2: Violin Plots, Correlation Matrix
            dbc.Row(
                [
                    # Column 1: Data Source Details
                    dbc.Col(
                        data_source_details,
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
