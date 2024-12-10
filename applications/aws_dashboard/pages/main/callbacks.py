"""Callbacks/Connections for the Main/Front Dashboard Page"""

from datetime import datetime
from dash import callback, Input, Output, ctx
from dash.exceptions import PreventUpdate

# SageWorks Imports
from sageworks.web_interface.page_views.main_page import MainPage
from sageworks.web_interface.components.plugins.ag_table import AGTable


# Update the last updated time
def last_updated():
    @callback(
        Output("data-last-updated", "children"),
        Input("main_page_refresh", "n_intervals"),
    )
    def refresh_last_updated_time(_n):
        # A string of the new time (in the local time zone)
        return datetime.now().strftime("Last Updated: %Y-%m-%d (%I:%M %p)")


# Update all of the artifact tables
def tables_refresh(main_page: MainPage, tables: dict[str, AGTable]):
    # Aggregate all the output properties for all the tables
    @callback(
        [
            Output(component_id, prop) for t in tables.values() for component_id, prop in t.properties
        ],  # Aggregate properties
        Input("main_page_refresh", "n_intervals"),
    )
    def _all_tables_update(_n):
        # Update all the properties for each table
        all_props = []

        # Data Sources
        data_sources = main_page.data_sources_summary()
        all_props.extend(tables["data_sources"].update_properties(data_sources))

        # Feature Sets
        feature_sets = main_page.feature_sets_summary()
        all_props.extend(tables["feature_sets"].update_properties(feature_sets))

        # Models
        models = main_page.models_summary()
        all_props.extend(tables["models"].update_properties(models))

        # Endpoints
        endpoints = main_page.endpoints_summary()
        all_props.extend(tables["endpoints"].update_properties(endpoints))

        # Return all the updated properties
        return all_props


def navigate_to_subpage(tables: dict[str, str]):
    @callback(
        Output("url", "href"),
        [Input(f"main_{table_name}", "selectedRows") for table_name in tables.keys()],
        prevent_initial_call=True,
    )
    def _navigate_to_subpage(*selected_rows_list):
        # Fixme: Let's circle back to this as some point
        trigger_table_name = ctx.triggered_id.replace("main_", "")  # Drop the "main_" prefix

        # Loop over each table_id and its selected rows
        for table_name, selected_rows in zip(tables.keys(), selected_rows_list):
            print(f"Trigger name: {trigger_table_name} Table Name: {table_name}, Selected Rows: {selected_rows}")
            if table_name == trigger_table_name and selected_rows:
                selected_row = selected_rows[0]
                row_uuid = selected_row.get("uuid", 0)
                subpage_name = table_name
                return f"/{subpage_name}?uuid={row_uuid}"

        raise PreventUpdate
