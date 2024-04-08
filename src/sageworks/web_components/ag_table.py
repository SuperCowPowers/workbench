"""An Example Table plugin component using AG Grid"""

import logging
import pandas as pd
from dash_ag_grid import AgGrid

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType

# Get the SageWorks logger
log = logging.getLogger("sageworks")


class AGTable(PluginInterface):
    """AGTable Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL_TABLE

    def create_component(self, component_id: str) -> AgGrid:
        """Create a Table Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            AgGrid: The Table Component using AG Grid
        """
        self.component_id = component_id
        self.container = AgGrid(
            id=component_id,
            # className="ag-theme-balham-dark",
            columnSize="sizeToFit",
            dashGridOptions={
                "rowHeight": None,
                "domLayout": "normal",
                "rowSelection": "single",
                "filter": True,
            },
            style={"maxHeight": "200px", "overflow": "auto"},

        )

        # Fill in content slots
        self.slots = [
            (self.component_id, "columnDefs"),
            (self.component_id, "rowData"),
            (self.component_id, "selectedRows"),
        ]

        # Output signals
        self.signals = [
            (self.component_id, "selectedRows"),
        ]

        # Return the container
        return self.container

    def update_contents(self, model_table: pd.DataFrame, **kwargs) -> list:
        """Update the contents for the plugin.

        Args:
            model_table (pd.DataFrame): A DataFrame with the model table data
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated contents (children)
        """
        log.important(f"Updating Table Plugin with a model table and kwargs: {kwargs}")

        # Convert the DataFrame to a list of dictionaries for AG Grid
        table_data = model_table.to_dict("records")

        # Define column definitions based on the DataFrame
        column_defs = [{"headerName": col, "field": col, "filter": "agTextColumnFilter"} for col in model_table.columns]

        # Select the first row by default
        selected_rows = model_table.head(1).to_dict("records")

        # Return the column definitions and table data (must match the content slots)
        return [column_defs, table_data, selected_rows]


if __name__ == "__main__":
    # Run the Unit Test for the Plugin
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(AGTable).run()
