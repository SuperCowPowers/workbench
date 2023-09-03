"""A table component"""
from dash import dash_table
from dash.dash_table.Format import Format
import plotly.express as px
import pandas as pd

# Local imports
from sageworks.web_components.color_maps import color_map_add_alpha


# Helper Functions
def column_setup(df: pd.DataFrame, show_columns: list[str] = None, markdown_columns: list[str] = None) -> list:
    """Internal: Get the column information for the given DataFrame
    Args:
        df: The DataFrame to get the column information
        show_columns: The columns to show
        markdown_columns: The columns to show as markdown
    Returns:
        list: The column information as a list of dicts
    """

    # HARDCODE: Not sure how to get around hard coding these columns
    dont_show = ["id", "uuid", "write_time", "api_invocation_time", "is_deleted", "x", "y"]

    # Only show these columns
    if not show_columns:
        show_columns = [c for c in df.columns.to_list() if c not in dont_show]

    # Column Setup with name, id, and presentation type
    column_setup_list = []
    for c in show_columns:
        presentation = "markdown" if markdown_columns and c in markdown_columns else "input"
        # Check for a numeric column
        if df[c].dtype in ["float64", "float32"]:
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


def style_data_conditional(color_column: str = None, unique_categories: list = None) -> list:
    """Internal: Style the cells based on the color column
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

    # To select rows we need to set up an (0->N) ID for each row
    df["id"] = range(len(df))

    # Column Setup with name, id, and presentation type
    column_setup_list = column_setup(df, show_columns, markdown_columns)

    # Construct our style_cell_conditionals
    if color_column is None or color_column not in df.columns:
        style_cells = style_data_conditional()
    else:
        unique_categories = df[color_column].unique().tolist()
        unique_categories = [x for x in unique_categories if x != "sample"]
        style_cells = style_data_conditional(color_column, unique_categories)

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
            "backgroundColor": "rgba(60, 60, 60, 0.5)",
            "hoverColor": "rgba(60, 60, 60, 0.5)",
            "color": "rgb(200, 200, 200)",
            "border": "0px",
        },
        tooltip_header=column_types,
        markdown_options={"html": True},
        style_header_conditional=[{"if": {"column_id": "del"}, "color": "transparent"}],
        style_cell_conditional=[
            {"if": {"column_id": "del"}, "padding": "7px 0px 0px 0px"},
        ],
        style_data_conditional=style_cells,
    )
    return table
