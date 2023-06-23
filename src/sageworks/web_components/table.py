"""A table component"""
from dash import dash_table
from dash.dash_table.Format import Format
import pandas as pd


# Helper Functions
def column_setup(df: pd.DataFrame,
                 show_columns: list[str] = None,
                 markdown_columns: list[str] = None) -> list:
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
        show_columns.remove("id")
        if "uuid" in show_columns:
            show_columns.remove("uuid")

    # Column Setup with name, id, and presentation type
    column_setup_list = []
    for c in show_columns:
        presentation = "markdown" if markdown_columns and c in markdown_columns else "input"
        # Check for a numeric column
        if df[c].dtype in ["float64", "float32"]:
            print(f"Column {c} is numeric")
            column_setup_list.append({"name": c, "id": c, "type": "numeric",
                                      "format": Format(group=",", precision=3, scheme="f")})
        else:
            column_setup_list.append({"name": c, "id": c, "presentation": presentation})
    return column_setup_list


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
) -> dash_table:
    """Create a Table"""

    # To select rows we need to set up an ID for each row
    df["id"] = df.index

    # Only show these columns
    if not show_columns:
        show_columns = df.columns.to_list()
        show_columns.remove("id")
        if "uuid" in show_columns:
            show_columns.remove("uuid")

    # Column Setup with name, id, and presentation type
    column_setup_list = column_setup(df, show_columns, markdown_columns)

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
            {"if": {"column_id": "remove"}, "width": "20px", "padding": "5px 0px 2px 0px", "overflow": "visible"}
        ],
        style_data_conditional=[
            {
                "if": {"state": "selected"},
                "backgroundColor": "inherit !important",
                "border": "inherit !important",
            }
        ]
    )
    return table
