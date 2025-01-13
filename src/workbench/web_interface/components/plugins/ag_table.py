"""An Example Table plugin component using AG Grid"""

import logging
import pandas as pd
from dash_ag_grid import AgGrid

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.symbols import tag_symbols

# Get the Workbench logger
log = logging.getLogger("workbench")


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
        grid_options = {
            "rowSelection": "single",
            "suppressCellFocus": True,
            "rowHeight": 25,
            "headerHeight": 30,
            "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
            "domLayout": "autoHeight",  # Automatically adjust height to fit content
        }

        self.container = AgGrid(
            id=component_id,
            dashGridOptions=grid_options,
            style={"maxHeight": f"{max_height}px", "overflow": "auto"},
        )

        # Fill in plugin properties
        self.properties = [(self.component_id, "columnDefs"), (self.component_id, "rowData")]

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
        log.info(f"Updating Table Plugin with a table dataframe and kwargs: {kwargs}")

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
                "width": 100 if col in ["Health", "Owner", "Ver"] else None,  # Smaller width for specific columns
                "cellStyle": {"fontSize": "18px"} if col == "Health" else None,  # Larger font for Health column
            }
            for col in table_df.columns
        ]

        # Return the column definitions and table data (must match the plugin properties)
        return [column_defs, table_data]


if __name__ == "__main__":
    # Run the Unit Test for the Plugin
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(AGTable, theme="quartz").run()
