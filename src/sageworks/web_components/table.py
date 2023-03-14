"""A table component"""
from dash import dash_table
import pandas as pd


def create(table_id: str, df: pd.DataFrame, show_columns: list[str] = None, row_select='single') -> dash_table:
    """Create a Table"""

    # To select rows we need to set up an ID for each row
    df['id'] = df.index

    # Only show these columns
    if not show_columns:
        show_columns = df.columns.to_list()
        show_columns.remove('id')

    table = dash_table.DataTable(
        id=table_id,
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns if i in show_columns],
        # style_as_list_view=True,
        sort_action='native',
        row_selectable=row_select,
        cell_selectable=False,
        # selected_rows=[0],
        # fixed_rows={'headers': True},
        style_table={'maxHeight': '300px', 'overflowY': 'auto'}
    )
    return table


# Storage
"""
style_header={
    'backgroundColor': 'rgb(120, 170, 120)',
    'color': 'rgb(20, 20, 20)'
},
style_data={
    'backgroundColor': 'rgb(30, 30, 30)',
    'color': 'rgb(180, 180, 220)'
}
"""
