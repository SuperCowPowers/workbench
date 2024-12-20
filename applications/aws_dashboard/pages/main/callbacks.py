"""Callbacks/Connections for the Main/Front Dashboard Page"""

from datetime import datetime
from dash import callback, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.web_interface.page_views.main_page import MainPage
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.utils.pandas_utils import dataframe_delta


# Update the last updated time
def last_updated():
    @callback(
        Output("data-last-updated", "children"),
        Input("main_page_refresh", "n_intervals"),
    )
    def refresh_last_updated_time(_n):
        # A string of the new time (in the local time zone)
        return datetime.now().strftime("%Y-%m-%d (%I:%M %p)")


def plugin_page_info():
    @callback(
        Output("plugin-pages", "children"),  # Update the entire Div
        Input("plugin-pages-info", "data"),  # Store holds plugin data
    )
    def render_plugin_pages(data):
        if not data:
            return html.Div(
                [
                    html.H4("Plugin Pages", style={"textAlign": "left"}),
                    html.A(
                        "Make some plugins! :)",
                        href="https://supercowpowers.github.io/workbench/plugins/",
                        target="_blank",  # Open link in a new tab
                        style={
                            "textDecoration": "none",  # Remove underline
                            "marginLeft": "20px",  # Add inset
                        },
                    ),
                ]
            )

        # Generate the list of plugin links
        plugin_list = html.Ul([html.Li(html.A(name, href=path)) for path, name in data.items()])

        # Return the header and list
        return html.Div(
            [
                html.H4("Plugin Pages", style={"textAlign": "left"}),
                plugin_list,
            ]
        )


# Update all of the artifact tables
def tables_refresh(main_page: MainPage, tables: dict[str, AGTable]):
    @callback(
        [Output(component_id, prop) for t in tables.values() for component_id, prop in t.properties]
        + [Output("table_hashes", "data")],  # Add hash updates
        [
            Input("main_page_refresh", "n_intervals"),
            State("table_hashes", "data"),  # Get current hashes
        ],
    )
    def _all_tables_update(_n, current_hashes):
        # Grab all tables and compute deltas
        updated_dataframes = {
            "data_sources": dataframe_delta(main_page.data_sources_summary, current_hashes["data_sources"]),
            "feature_sets": dataframe_delta(main_page.feature_sets_summary, current_hashes["feature_sets"]),
            "models": dataframe_delta(main_page.models_summary, current_hashes["models"]),
            "endpoints": dataframe_delta(main_page.endpoints_summary, current_hashes["endpoints"]),
        }

        # Check if all DataFrames are None (no changes)
        if all(df is None for df, _ in updated_dataframes.values()):
            raise PreventUpdate

        # Initialize list for all updated properties and new hashes
        all_props = []
        new_hashes = current_hashes.copy()

        # Update properties for each table
        for key, (df, hash_value) in updated_dataframes.items():
            if df is None:
                # No changes, use no_update for placeholders
                all_props.extend([no_update] * len(tables[key].properties))
            else:
                # Table has changes, update properties
                all_props.extend(tables[key].update_properties(df))
                new_hashes[key] = hash_value

        # Return updated properties and the new hash data
        return all_props + [new_hashes]


# Navigate to the subpages
def navigate_to_subpage():
    @callback(
        Output("url", "href", allow_duplicate=True),
        Input("main_data_sources", "selectedRows"),
        prevent_initial_call=True,
    )
    def navigate_data_sources(selected_rows):
        if selected_rows:
            row_uuid = selected_rows[0].get("uuid", 0)
            return f"/data_sources?uuid={row_uuid}"
        raise PreventUpdate

    @callback(
        Output("url", "href", allow_duplicate=True),
        Input("main_feature_sets", "selectedRows"),
        prevent_initial_call=True,
    )
    def navigate_feature_sets(selected_rows):
        if selected_rows:
            row_uuid = selected_rows[0].get("uuid", 0)
            return f"/feature_sets?uuid={row_uuid}"
        raise PreventUpdate

    @callback(
        Output("url", "href", allow_duplicate=True),
        Input("main_models", "selectedRows"),
        prevent_initial_call=True,
    )
    def navigate_models(selected_rows):
        if selected_rows:
            row_uuid = selected_rows[0].get("uuid", 0)
            return f"/models?uuid={row_uuid}"
        raise PreventUpdate

    @callback(
        Output("url", "href", allow_duplicate=True),
        Input("main_endpoints", "selectedRows"),
        prevent_initial_call=True,
    )
    def navigate_predictions(selected_rows):
        if selected_rows:
            row_uuid = selected_rows[0].get("uuid", 0)
            return f"/endpoints?uuid={row_uuid}"
        raise PreventUpdate
