import logging
import pandas as pd
from dash import dash_table

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType

# Get the SageWorks logger
log = logging.getLogger("sageworks")


class DataTable(PluginInterface):
    """DataTable Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL_TABLE

    def create_component(self, component_id: str) -> dash_table.DataTable:
        """Create a Table Component without any data."""
        self.component_id = component_id
        self.container = dash_table.DataTable(
            id=component_id,
            columns=[],
            data=[],
            filter_action="native",  # Enable filtering
            sort_action="native",  # Enable sorting
            row_selectable="single",  # Enable single row selection
            selected_rows=[0],  # Select the first row by default
            style_table={"maxHeight": "200px", "overflow": "auto"},  # Table styling
        )

        # Fill in plugin properties
        self.properties = [
            (self.component_id, "columns"),
            (self.component_id, "data"),
        ]

        # Output signals
        self.signals = [
            (self.component_id, "selected_rows"),
        ]

        return self.container

    def update_properties(self, model_table: pd.DataFrame, **kwargs) -> list:
        """Update the properties for the plugin."""
        log.important(f"Updating DataTable Plugin with a model table and kwargs: {kwargs}")

        # Convert the DataFrame to a list of dictionaries for DataTable
        table_data = model_table.to_dict("records")

        # Define column definitions based on the DataFrame
        columns = [{"name": col, "id": col} for col in model_table.columns]

        # Return the column definitions and table data (must match the plugin properties)
        return [columns, table_data]


if __name__ == "__main__":
    # Run the Unit Test for the Plugin
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(DataTable).run()
