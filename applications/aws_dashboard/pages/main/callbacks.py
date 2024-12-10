"""Callbacks/Connections for the Main/Front Dashboard Page"""

from datetime import datetime
from dash import callback, Input, Output
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
