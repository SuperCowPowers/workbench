"""A table component"""
from dash import dash_table
import pandas as pd


def create(
    table_id: str,
    df: pd.DataFrame,
    header_color="rgb(60, 60, 60)",
    show_columns: list[str] = None,
    row_select=False,
    markdown_columns: list[str] = None,
    max_height: str = "200px",
    columns_editable: bool = False,
    fixed_headers: bool = False,
) -> dash_table:
    """Create a Table"""

    # To select rows we need to set up an ID for each row
    df["id"] = df.index

    # Only show these columns
    if not show_columns:
        show_columns = df.columns.to_list()
        show_columns.remove("id")

    # Column Setup with name, id, and presentation type
    column_setup = []
    for c in show_columns:
        presentation = "markdown" if markdown_columns and c in markdown_columns else "input"
        if columns_editable:
            column_setup.append({"name": c, "id": c, "deletable": True, "selectable": True})
        else:
            column_setup.append({"name": c, "id": c, "presentation": presentation})

    # Create the Dash Table
    table = dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=column_setup,
        sort_action="native",
        row_selectable=row_select,
        cell_selectable=False,
        selected_rows=[0],
        fixed_rows={'headers': fixed_headers},
        style_table={"maxHeight": max_height, "overflowX": "auto", "overflowY": "auto"},
        style_as_list_view=True,
        style_cell={
            "font-family": "HelveticaNeue",
            "padding": "5px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "maxWidth": 250
        },
        style_header={
            "textAlign": "left",
            "fontSize": 16,
            "backgroundColor": header_color,
            "color": "rgb(200, 200, 200)"
        },
        style_data={
            "fontSize": 14,
            "backgroundColor": "rgb(60, 60, 60)",
            "color": "rgb(200, 200, 200)",
            "border": "0px"
        },
        markdown_options={"html": True},
    )
    return table
