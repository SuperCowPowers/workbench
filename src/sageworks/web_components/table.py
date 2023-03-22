"""A table component"""
from dash import dash_table
import pandas as pd


def create(table_id: str, df: pd.DataFrame, header_color='rgb(60, 60, 60)',
           show_columns: list[str] = None, row_select=False,
           markdown_columns: list[str] = None) -> dash_table:
    """Create a Table"""

    # To select rows we need to set up an ID for each row
    df['id'] = df.index

    # Only show these columns
    if not show_columns:
        show_columns = df.columns.to_list()
        show_columns.remove('id')

    # Column Setup with name, id, and presentation type
    column_setup = []
    for c in show_columns:
        presentation = 'markdown' if markdown_columns and c in markdown_columns else 'input'
        column_setup.append({"name": c, "id": c, "presentation": presentation})

    # Create the Dash Table
    table = dash_table.DataTable(
        id=table_id,
        data=df.to_dict('records'),
        columns=column_setup,
        # style_as_list_view=True,
        sort_action='native',
        row_selectable=row_select,
        cell_selectable=False,
        # selected_rows=[0],
        # fixed_rows={'headers': True},
        style_table={'maxHeight': '300px', 'overflowY': 'auto'},
        style_as_list_view=True,
        style_cell={'font-family': 'HelveticaNeue', 'padding': '10px'},
        style_header={
            'fontSize': 18,
            'backgroundColor': header_color,
            'color': 'rgb(200, 200, 200)'
        },
        style_data={
            'fontSize': 18,
            'backgroundColor': 'rgb(80, 80, 80)',
            'color': 'rgb(200, 200, 200)'
        },
        markdown_options={"html": True}
    )
    return table


# Storage
"""

"""
