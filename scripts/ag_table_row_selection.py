import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
from dash_ag_grid import AgGrid

app = dash.Dash(__name__)

# Placeholder DataFrame
df = pd.DataFrame()

app.layout = html.Div(
    [
        html.Button("Update Table", id="update-button", n_clicks=0),
        AgGrid(
            id="my-table",
            columnSize="sizeToFit",
            dashGridOptions={
                "rowHeight": None,
                "domLayout": "normal",
                "rowSelection": "single",
                "filter": True,
            },
            style={"maxHeight": "200px", "overflow": "auto"},
        ),
        html.Div(id="selected-row-info"),  # Div to display selected row information
    ]
)


@app.callback(
    Output("my-table", "columnDefs"),
    Output("my-table", "rowData"),
    Output("my-table", "selectedRows"),
    Input("update-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_data(n_clicks):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    column_defs = [{"headerName": col, "field": col, "filter": "agTextColumnFilter"} for col in df.columns]
    row_data = df.to_dict("records")
    selected_rows = df.head(1).to_dict("records")
    selected_rows = {"ids": ["1"]}
    print("Updating Table")
    # return column_defs, row_data, selected_rows
    return (
        column_defs,
        row_data,
        selected_rows,
    )


@app.callback(Output("selected-row-info", "children"), Input("my-table", "selectedRows"))
def update_selected_row_info(selected_rows):
    selected_row_info = f"Selected Rows: {selected_rows}"
    print(selected_row_info)
    return selected_row_info


if __name__ == "__main__":
    app.run_server(debug=True)
