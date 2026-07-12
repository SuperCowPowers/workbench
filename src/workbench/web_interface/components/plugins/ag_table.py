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
    header_height = 36
    row_height = 25
    # Thin columns: small by default (auto-size cap) but user can drag wider
    thin_columns = {"Health": 100, "Owner": 100}

    def create_component(self, component_id: str, max_height: int = 500, row_selection: str = "single") -> AgGrid:
        """Create a Table Component without any data.

        Args:
            component_id (str): The ID of the component
            max_height (int): Maximum height in pixels (default: 500)
            row_selection (str): Row selection mode - "single" or "multiple" (default: "single")

        Returns:
            AgGrid: The AG Grid component
        """
        self.component_id = component_id
        self.max_height = max_height

        # AG Grid configuration for tighter rows and columns
        column_limits = [{"colId": col, "maxWidth": w} for col, w in self.thin_columns.items()]
        grid_options = {
            "rowSelection": row_selection,
            "suppressCellFocus": True,
            "rowHeight": self.row_height,
            "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
            "autoSizeStrategy": {
                "type": "fitCellContents",
                "defaultMaxWidth": 400,
                "columnLimits": column_limits,
            },
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
            table_df["Health"] = table_df["Health"].map(tag_symbols)

        # Convert the DataFrame to a list of dictionaries for AG Grid
        table_data = table_df.to_dict("records")

        # Create column definitions for AG Grid
        column_defs = [{"headerName": col, "field": col} for col in table_df.columns]

        # Dynamically adjust height based on row count
        row_count = len(table_df)
        computed_height = min(self.header_height + self.row_height * row_count, self.max_height) + 2
        style = {"height": f"{computed_height}px", "overflow": "auto"}

        # Return the column definitions, table data, and style (must match the plugin properties)
        return [column_defs, table_data, style]


if __name__ == "__main__":
    # Run the Unit Test for the Plugin
    from workbench.api import Meta
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Test on model data
    models_df = Meta().models(details=True)

    # Run the Unit Test on the Plugin
    PluginUnitTest(AGTable, theme="dark", input_data=models_df, max_height=500).run()
