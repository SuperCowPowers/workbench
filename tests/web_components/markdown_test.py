from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.web_components.data_details_markdown import DataDetailsMarkdown
from sageworks.views.data_source_web_view import DataSourceWebView

# Initialize your component class
component_instance = DataDetailsMarkdown()

# Initialize the component. Add any additional args or kwargs you need
markdown_component = component_instance.create_component(component_id="data_details_markdown")

# Create a simple Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
)

# Add the component and a timer to the layout
app.layout = html.Div(
    [markdown_component, dcc.Interval(id="interval-component", interval=5 * 1000, n_intervals=0)]  # in milliseconds
)

# Create a DataSourceWebView
data_source_web_view = DataSourceWebView()


# Create a callback to update the component
@app.callback(
    Output("data_details_markdown", "children"), Input("interval-component", "n_intervals"), prevent_initial_call=True
)
def update_details(n):
    data_details = data_source_web_view.data_source_details(0)
    markdown_str = component_instance.generate_markdown(data_details)
    return markdown_str


"""Run the app"""
if __name__ == "__main__":
    app.run_server(debug=True)
