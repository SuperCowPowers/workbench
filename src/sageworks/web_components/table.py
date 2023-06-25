"""A table component"""
from dash import dash_table
from dash.dash_table.Format import Format
import plotly.express as px
import pandas as pd


# Helper Functions
def column_setup(
    df: pd.DataFrame, show_columns: list[str] = None, markdown_columns: list[str] = None
) -> list:
    """Internal: Get the column information for the given DataFrame
    Args:
        df: The DataFrame to get the column information
        show_columns: The columns to show
        markdown_columns: The columns to show as markdown
    Returns:
        list: The column information as a list of dicts
    """
    # Only show these columns
    if not show_columns:
        show_columns = df.columns.to_list()
        if "id" in show_columns:
            show_columns.remove("id")
        if "uuid" in show_columns:
            show_columns.remove("uuid")
        if "x" in show_columns:
            show_columns.remove("x")
        if "y" in show_columns:
            show_columns.remove("y")

    # Column Setup with name, id, and presentation type
    column_setup_list = []
    for c in show_columns:
        presentation = (
            "markdown" if markdown_columns and c in markdown_columns else "input"
        )
        # Check for a numeric column
        if df[c].dtype in ["float64", "float32"]:
            print(f"Column {c} is numeric")
            column_setup_list.append(
                {
                    "name": c,
                    "id": c,
                    "type": "numeric",
                    "format": Format(group=",", precision=3, scheme="f"),
                }
            )
        else:
            column_setup_list.append({"name": c, "id": c, "presentation": presentation})
    return column_setup_list


def hex_to_rgb(hex_color: str) -> str:
    """Internal: Convert a hex color to rgb
    Args:
        hex_color: The hex color to convert
    Returns:
        str: The rgb color
    """
    hex_color = hex_color.lstrip("#")
    rgb = f"rgba({int(hex_color[:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:], 16)}, 0.25)"
    return rgb


def color_map_setup(color_map: list) -> list:
    """Internal: Add alpha to the given color map
    Args:
        color_map: The color map to add alpha to
    Returns:
        list: The color map with alpha added
    """
    return [hex_to_rgb(color_map[i]) for i in range(len(color_map))]


def style_data_conditional(color_column: str = None) -> list:
    """Internal: Style the cells based on the color column
    Args:
        color_column: The column to use for the cell color
    Returns:
        list: The cell style information as a list of dicts
    """

    # This just make a selected cell 'transparent' so it doesn't look selected
    style_cells = [
        {
            "if": {"state": "selected"},
            "backgroundColor": "inherit !important",
            "border": "inherit !important",
        }
    ]

    # If they want to color the cells based on a column value (like cluster)
    if color_column is not None:
        hex_color_map = px.colors.qualitative.Plotly
        len_color_map = len(hex_color_map)
        color_map = color_map_setup(hex_color_map)
        style_cells += [
            {
                "if": {
                    "filter_query": f"{{cluster}} = {lookup}"
                },
                "backgroundColor": f"{color_map[lookup % len_color_map]}"
            } for lookup in range(len_color_map)
        ]

    return style_cells


def create(
    table_id: str,
    df: pd.DataFrame,
    column_types: dict = None,
    header_color="rgb(60, 60, 60)",
    show_columns: list[str] = None,
    row_select=False,
    markdown_columns: list[str] = None,
    max_height: str = "200px",
    fixed_headers: bool = False,
    color_column: str = None,
) -> dash_table:
    """Create a Table"""

    # To select rows we need to set up an ID for each row
    df["id"] = df.index

    # Column Setup with name, id, and presentation type
    column_setup_list = column_setup(df, show_columns, markdown_columns)

    # Construct our style_cell_conditionals
    style_cells = style_data_conditional(color_column)

    # Create the Dash Table
    table = dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=column_setup_list,
        sort_action="native",
        row_selectable=row_select,
        cell_selectable=True,
        selected_rows=[0],
        fixed_rows={"headers": fixed_headers},
        style_table={"maxHeight": max_height, "overflowX": "auto", "overflowY": "auto"},
        style_as_list_view=True,
        style_cell={
            "font-family": "HelveticaNeue",
            "padding": "5px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "maxWidth": 250,
            "textAlign": "left",
        },
        style_header={
            "textAlign": "left",
            "fontSize": 16,
            "backgroundColor": header_color,
            "color": "rgb(200, 200, 200)",
        },
        style_data={
            "fontSize": 14,
            "backgroundColor": "rgb(60, 60, 60)",
            "color": "rgb(200, 200, 200)",
            "border": "0px",
        },
        tooltip_header=column_types,
        markdown_options={"html": True},
        style_header_conditional=[
            {"if": {"column_id": "remove"}, "color": "transparent"}
        ],
        style_cell_conditional=[
            {"if": {"column_id": "remove"}, "width": "10px", "padding": "0px 0px 0px 0px"},
        ],
        style_data_conditional=style_cells,
    )
    return table
