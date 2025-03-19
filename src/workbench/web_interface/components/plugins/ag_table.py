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
    max_height = 500
    header_height = 30
    row_height = 25

    def create_component(
        self, component_id: str, header_color: str = "rgb(120, 60, 60)", max_height: int = 500
    ) -> AgGrid:
        """Create a Table Component without any data."""
        self.component_id = component_id
        self.max_height = max_height

        # AG Grid configuration for tighter rows and columns
        grid_options = {
            "rowSelection": "single",
            "suppressCellFocus": True,
            "headerHeight": self.header_height,
            "rowHeight": self.row_height,
            "defaultColDef": {"sortable": True, "filter": True, "resizable": True, "maxWidth": 500},
            "autoSizeStrategy": {"type": "fitCellContents"},
        }

        self.container = AgGrid(
            id=component_id,
            dashGridOptions=grid_options,
            style={"height": f"{self.max_height}px", "overflow": "auto"},
        )

        # Fill in plugin properties
        self.properties = [
            (self.component_id, "columnDefs"),
            (self.component_id, "rowData"),
            (self.component_id, "style"),
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
        log.info(f"Updating Table Plugin with a table dataframe and kwargs: {kwargs}")

        # Add Health Symbols
        if "Health" in table_df.columns:
            table_df["Health"] = table_df["Health"].map(lambda x: tag_symbols(x))

        # Convert the DataFrame to a list of dictionaries for AG Grid
        table_data = table_df.to_dict("records")

        # Create column definitions for AG Grid
        column_defs = [
            {
                "headerName": col,
                "field": col,
            }
            for col in table_df.columns
        ]

        # Dynamically adjust height based on row count
        row_count = len(table_df)
        computed_height = min(self.header_height + self.row_height * row_count, self.max_height) + 2
        style = {"height": f"{computed_height}px", "overflow": "auto"}

        # Return the column definitions, table data, and style (must match the plugin properties)
        return [column_defs, table_data, style]


if __name__ == "__main__":
    # Run the Unit Test for the Plugin
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Test data
    data = {
        "ID": [f"id_{i}" for i in range(10)],
        "feat1": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat3": [0.1, 0.15, 0.2, 0.9, 2.8, 0.25, 0.35, 0.4, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
        "name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "Z" * 55],
    }
    test_df = pd.DataFrame(data)

    # Run the Unit Test on the Plugin
    PluginUnitTest(AGTable, theme="quartz", input_data=test_df, max_height=500).run()
