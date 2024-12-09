"""Callbacks/Connections for the Main/Front Dashboard Page"""

from datetime import datetime
from dash import callback, Input, Output, callback_context
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


def navigate_to_subpage(tables: dict[str, AGTable]):
    """Setup navigation callbacks for all the tables."""

    @callback(
        [Output("navigate-link", "href"), Output("navigate-link", "children")],
        [Input(f"main_{table_id}", "selectedRows") for table_id in tables.keys()],
        prevent_initial_call=True,
    )
    def _navigate_to_subpage(*selected_rows_list):
        # Identify which table triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        # Get the triggering table's ID
        triggered_table = ctx.triggered[0]["prop_id"].split(".")[0]

        # Map the triggered table ID to its corresponding subpage
        for subpage_name, table_id in zip(tables.keys(), tables.keys()):
            if f"main_{table_id}" == triggered_table:
                selected_rows = selected_rows_list[list(tables.keys()).index(table_id)]
                if selected_rows:  # Check if rows are selected
                    # Construct the target URL for the selected row
                    return f"/{subpage_name}", "click"  # Update href and trigger click

        # No selection made in any table
        raise PreventUpdate
