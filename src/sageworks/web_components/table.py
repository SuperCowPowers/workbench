"""A table component"""
from dash import dash_table
import pandas as pd


def create(table_id: str, df: pd.DataFrame, show_columns: list[str] = None, row_select='single') -> dash_table:
    """Create a Table"""

    # To select rows we need to set up an ID for each row
    df['id'] = df.index

    # Silly header color logic
    color_lookup = {'INCOMING_DATA': 'rgb(60, 60, 100)',
                    'DATA_SOURCES': 'rgb(100, 60, 60)',
                    'FEATURE_SETS': 'rgb(100, 100, 60)',
                    'MODELS': 'rgb(60, 100, 60)',
                    'ENDPOINTS': 'rgb(100, 60, 100)'}
    header_color = color_lookup.get(table_id, 'rgb(60, 60, 60)')

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
        # row_selectable=row_select,
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
        }
    )
    return table


# Storage
"""

"""
