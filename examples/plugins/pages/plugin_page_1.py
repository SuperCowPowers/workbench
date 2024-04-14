"""Plugin Page 1:  A 'Hello World' SageWorks Plugin Page"""

import dash
from dash import html, page_container, register_page
import dash_bootstrap_components as dbc


class PluginPage1:
    """Plugin Page:  A SageWorks Plugin Web Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Hello World"

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Register this page with Dash and set up the layout (required)
        register_page(
            __file__,
            path="/plugin_1",
            name=self.page_name,
            layout=self.page_layout(),
        )

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        layout = dash.html.Div(children=[dash.html.H1(self.page_name)])
        return layout


# Unit Test for your Plugin Page
if __name__ == "__main__":
    import webbrowser

    # Create our Dash Application
    my_app = dash.Dash(
        __name__,
        title="SageWorks Dashboard",
        use_pages=True,
        pages_folder="",
        external_stylesheets=[dbc.themes.DARKLY],
    )

    # For Multi-Page Applications, we need to create a 'page container' to hold all the pages
    my_app.layout = html.Div([page_container])

    # Create the Plugin Page and call page_setup
    plugin_page = PluginPage1()
    plugin_page.page_setup(my_app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/plugin_1")

    # Note: This 'main' is purely for running/testing locally
    my_app.run(host="localhost", port=8000, debug=True)
