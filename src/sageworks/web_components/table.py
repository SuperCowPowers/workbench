"""A table component"""

from dash import dash_table
from dash.dash_table.Format import Format
import plotly.express as px
import pandas as pd

# Local imports
from sageworks.web_components.color_maps import color_map_add_alpha

# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface


class Table(ComponentInterface):
    """Data Details Markdown Component"""

    def create_component(
        self,
        component_id: str,
        header_color="rgb(60, 60, 60)",
        row_select=False,
        max_height: int = 200,
        fixed_headers: bool = True,
        **kwargs,
    ) -> dash_table.DataTable:
        """Create a DataTable Component without any data.
        Args:
            component_id (str): The ID of the web component
            header_color: The color of the table header bar
            row_select (bool): If the table should be row selectable
            max_height (str): The maximum height of the table
            fixed_headers (bool): If the headers should be fixed
        Returns:
            dash_table.DataTable: A Dash DataTable Component
        """
        table = dash_table.DataTable(
            id=component_id,
            columns=[{"name": "Status", "id": "status"}],
            data=[{"status": "Waiting for data..."}],
            sort_action="native",
            row_selectable=row_select,
            cell_selectable=False,
            selected_rows=[0],
            fixed_rows={"headers": fixed_headers},
            style_table={"maxHeight": max_height, "overflowX": "auto", "overflowY": "auto"},
            style_as_list_view=True,
            style_cell={
                "font-family": "HelveticaNeue",
                "padding": "5px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "minWidth": 80,
                "maxWidth": 250,
                "textAlign": "left",
            },
            style_header={
                "className": "dash-table-header",
                "textAlign": "left",
                "fontSize": 14,
                "padding": "5px 5px 5px 5px",
                "backgroundColor": header_color,
                "backgroundImage": f"linear-gradient(to bottom, {header_color}, rgba(0, 0, 0, 0.5))",
                "color": "rgb(200, 200, 200)",
                "border": "0px",
            },
            style_data={
                "fontSize": 13,
                "backgroundColor": "rgba(60, 60, 60, 0.5)",
                "color": "rgb(200, 200, 200)",
                "border": "0px",
            },
            style_data_conditional=[{"if": {"column_id": "Health"}, "fontSize": 16, "textAlign": "left"}],
            markdown_options={"html": True},
            **kwargs,
        )
        return table

    # Helper Functions
    @staticmethod
    def column_setup(df: pd.DataFrame, show_columns: list[str] = None, markdown_columns: list[str] = None) -> list:
        """Get the column information for the given DataFrame
        Args:
            df: The DataFrame to get the column information
            show_columns: The columns to show
            markdown_columns: The columns to show as markdown
        Returns:
            list: The column information as a list of dicts
        """

        # HARDCODE: Not sure how to get around hard coding these columns
        dont_show = [
            "id",
            "uuid",
            "write_time",
            "event_time",
            "api_invocation_time",
            "is_deleted",
            "x",
            "y",
        ]

        # Only show these columns
        if not show_columns:
            show_columns = [c for c in df.columns.to_list() if c not in dont_show]

        column_setup_list = []
        for c in show_columns:
            column_def = {
                "name": c,
                "id": c,
                "presentation": "markdown" if markdown_columns and c in markdown_columns else "input",
            }

            # Check for a numeric column and add additional properties as needed
            if df[c].dtype in ["float64", "float32"]:
                column_def.update(
                    {
                        "type": "numeric",
                        "format": Format(group=",", precision=3, scheme="f"),
                    }
                )

            column_setup_list.append(column_def)

        return column_setup_list

    @staticmethod
    def style_data_conditional(color_column: str = None, unique_categories: list = None) -> list:
        """Style the cells based on the color column
        Args:
            color_column: The column to use for the cell color
            unique_categories: The unique categories (string) in the color_column
        Returns:
            list: The cell style information as a list of dicts
        """

        # This just makes a selected cell 'transparent' so it doesn't look selected
        style_cells = [
            {
                "if": {"state": "selected"},
                "backgroundColor": "inherit !important",
                "border": "inherit !important",
            }
        ]

        # If they want to color the cells based on a column value
        if color_column is not None and unique_categories is not None:
            hex_color_map = px.colors.qualitative.Plotly
            len_color_map = len(hex_color_map)
            color_map = color_map_add_alpha(hex_color_map, 0.15)
            style_cells += [
                {
                    "if": {"filter_query": f"{{{color_column}}} = {cat}"},
                    "backgroundColor": f"{color_map[i % len_color_map]}",
                }
                for i, cat in enumerate(unique_categories)
            ]

        return style_cells


if __name__ == "__main__":
    from dash import Dash, html
    from sageworks.api.data_source import DataSource

    # Get a smart sample from a test data source
    ds = DataSource("abalone_data")
    df = ds.smart_sample()

    app = Dash(__name__)
    my_table = Table()
    table_comp = my_table.create_component("test-table")

    # Testing HTML Links
    columns = df.columns.to_list()
    df["Name"] = df.index
    df = df[["Name"] + columns]
    df["Name"] = df["Name"].map(lambda x: f"<a href='https://www.google.com' target='_blank'>{x}</a>")

    table_comp.columns = my_table.column_setup(df, markdown_columns=["Name"])
    table_comp.data = df.to_dict("records")
    app.layout = html.Div([table_comp])

    if __name__ == "__main__":
        app.run(debug=True)
