"""An Example Table plugin component using AG Grid"""

import logging
import pandas as pd
from dash_ag_grid import AgGrid

# SageWorks Imports
from sageworks.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.utils.symbols import tag_symbols

# Get the SageWorks logger
log = logging.getLogger("sageworks")


class AGTable(PluginInterface):
    """AGTable Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def create_component(
        self, component_id: str, header_color: str = "rgb(120, 60, 60)", max_height: int = 800
    ) -> AgGrid:
        """Create a Table Component without any data."""
        self.component_id = component_id

        # AG Grid configuration for tighter rows and columns
        grid_options = {"rowSelection": "single", "rowHeight": 30, "headerHeight": 40}

        self.container = AgGrid(
            id=component_id,
            dashGridOptions=grid_options,
            style={"maxHeight": f"{max_height}px", "overflow": "auto"},
        )

        # Fill in plugin properties
        self.properties = [
            (self.component_id, "columnDefs"),
            (self.component_id, "rowData"),
            (self.component_id, "selectedRows"),
        ]

        # Output signals
        self.signals = [
            (self.component_id, "selectedRows"),
        ]

        return self.container

    def update_properties(self, table_df: pd.DataFrame, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            table_df (pd.DataFrame): A DataFrame with the table data
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Table Plugin with a table dataframe and kwargs: {kwargs}")

        # Add Health Symbols
        if "Health" in table_df.columns:
            table_df["Health"] = table_df["Health"].map(lambda x: tag_symbols(x))

        # Convert the DataFrame to a list of dictionaries for AG Grid
        table_data = table_df.to_dict("records")

        # Okay the Health and Owner columns are always way too big
        column_defs = [
            {
                "headerName": col,
                "field": col,
                "resizable": True,
                "width": 80 if col in ["Health", "Owner", "Ver"] else None,  # Smaller width for specific columns
                "cellStyle": {"fontSize": "18px"} if col == "Health" else None,  # Larger font for Health column
            }
            for col in table_df.columns
        ]

        # Select the first row by default
        selected_rows = table_df.head(1).to_dict("records")
        print(f"SELECTED ROWS: {selected_rows}")

        # Return the column definitions and table data (must match the plugin properties)
        return [column_defs, table_data, selected_rows]


if __name__ == "__main__":
    # Run the Unit Test for the Plugin
    from sageworks.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(AGTable, theme="quartz").run()
