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


# Update the all of the artifact tables
def tables_refresh(main_page: MainPage, tables: dict[str, AGTable]):
    # Aggregate all the output properties for all the tables
    @callback(
        [Output(component_id, prop) for t in tables.values() for component_id, prop in t.properties],  # Aggregate properties
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


def setup_navigation_models():
    @callback(
        Output("url", "pathname"),  # Update the URL path
        Input("main_models", "selectedRows"),  # Trigger on row selection
        prevent_initial_call=True,
    )
    def navigate_to_models_page(selected_rows):
        if not selected_rows:
            raise PreventUpdate
        selected_model_id = selected_rows[0]["uuid"]
        return f"/models"
        # return f"/models/{selected_model_id}"  # Navigate to the specific model subpage
