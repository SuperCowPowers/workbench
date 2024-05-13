import dash
from dash import dcc, html, Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='theme-selector',
        options=[
            {'label': 'Light Theme', 'value': 'scatter-plot-light'},
            {'label': 'Dark Theme', 'value': 'scatter-plot-dark'}
        ],
        value='scatter-plot-light'  # Default value
    ),
    dcc.Graph(
        id='scatter-plot',
        className='scatter-plot-light',  # Initial class name
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'scatter', 'name': 'SF'},
            ],
            'layout': {
                'title': 'Scatter Plot Example'
            }
        }
    )
])

# Callback to update the className of the scatter plot based on theme selection
@app.callback(
    Output('scatter-plot', 'className'),
    Input('theme-selector', 'value')
)
def update_scatter_plot_class(selected_theme):
    print(selected_theme)
    return selected_theme

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

