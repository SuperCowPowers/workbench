import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np

# Generate random data
x_data = np.random.rand(100)
y_data = np.random.rand(100) * 12

# Create the scatter plot
fig = go.Figure(data=go.Scattergl(x=x_data, y=y_data, mode="markers"))

# Fine-tuning of the plot
fig.update_layout(
    margin={"t": 40, "b": 40, "r": 40, "l": 40, "pad": 0},
    xaxis=dict(title="X Axis", tickformat=".2f"),
    yaxis=dict(title="Y Axis", tickformat=".2f"),
    showlegend=False,
    dragmode="pan",
)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([dcc.Graph(id="scatter-plot", figure=fig, config={"scrollZoom": True})])

if __name__ == "__main__":
    app.run_server(debug=True)
