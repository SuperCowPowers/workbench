"""A table component testing script
   Note: This is purely slash/hack code to test and compare Table components
         When the components are finalized, the classes will go into sageworks/web_components
"""

import sys
from dash import dash_table
from dash.dash_table.Format import Format
import plotly.express as px
import pandas as pd
from dash_ag_grid import AgGrid

# SageWorks Imports
try:
    from sageworks.web_components.table import Table
    from sageworks.utils.symbols import health_icons
except ImportError:
    print("Please install sageworks")
    print("pip install sageworks")
    sys.exit(1)


# Just smashing the AgGrid class into this script for testing
class AGTable:
    """AGTable Component"""

    def __init__(self):
        self.component_id = None
        self.container = None
        self.properties = []
        self.signals = []

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
            className="ag-theme-alpine-auto-dark",
            columnSize="sizeToFit",
            dashGridOptions={"rowSelection": "single"},
            style={"maxHeight": "200px", "overflow": "auto"},
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

        # Return the container
        return self.container

    def update_properties(self, table_df: pd.DataFrame, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            table_df (pd.DataFrame): A DataFrame with the table data
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        print(f"Updating Table Plugin with a table dataframe and kwargs: {kwargs}")

        # Convert the DataFrame to a list of dictionaries for AG Grid
        table_data = table_df.to_dict("records")

        # Define column definitions based on the DataFrame
        column_defs = [{"headerName": col, "field": col, "filter": "agTextColumnFilter"} for col in table_df.columns]

        # Select the first row by default
        selected_rows = table_df.head(1).to_dict("records")

        # Return the column definitions and table data (must match the plugin properties)
        return [column_defs, table_data, selected_rows]


if __name__ == "__main__":
    from dash import Dash, html

    # A test dataframe
    health_list = ["healthy", "failed", "no_model", "5xx_errors", "no_endpoint", "model_type_unknown"]
    health_icons = [health_icons[h] for h in health_list]
    data = {
        "Name": ["joe", "bob", "sue", "jane", "jill", "jack"],
        "Health": health_icons,
        "Age": [10, 20, 30, 40, 50, 60],
        "Company": ["IBM", "Google", "Amazon", "Facebook", "Apple", "Microsoft"],
        "Title": ["CEO", "CFO", "CTO", "CIO", "COO", "CMO"],
        "Salary": [100, 200, 300, 400, 500, 600],
        "Bonus": [10, 20, 30, 40, 50, 60],
    }
    df = pd.DataFrame(data)

    # Testing HTML Links
    df["Company"] = df["Company"].map(lambda x: f"<a href='https://www.google.com' target='_blank'>{x}</a>")

    # Create a Dash app
    app = Dash(__name__)

    # Create the existing table component
    my_table = Table()
    existing_table = my_table.create_component(
        "current-table", header_color="rgb(60, 100, 60)", row_select="single", transparent=False
    )

    # Note: This is old style API, but will be replaced anyway
    existing_table.columns = my_table.column_setup(df, markdown_columns=["Company"])
    existing_table.data = df.to_dict("records")

    # Create the new AG Table component
    ag_table = AGTable()
    ag_table_component = ag_table.create_component("ag-table")

    # This would normally be a callback, but we're just testing
    ag_table_component.columnDefs, ag_table_component.rowData, ag_table_component.selectedRows = (
        ag_table.update_properties(df)
    )

    # Set up the layout
    app.layout = html.Div([existing_table, ag_table_component])

    if __name__ == "__main__":
        app.run(debug=True)
