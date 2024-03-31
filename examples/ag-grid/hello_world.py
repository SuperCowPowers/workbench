from dash import Dash, html
import dash_ag_grid as dag
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/wind_dataset.csv")

app = Dash(__name__)

columnDefs = [
    {"field": "direction"},
    {"field": "strength"},
    {"field": "frequency"},
]

grid = dag.AgGrid(
    id="get-started-example-basic",
    rowData=df.to_dict("records"),
    columnDefs=columnDefs,
)

app.layout = html.Div([grid])

if __name__ == "__main__":
    app.run(debug=True)
