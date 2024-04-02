"""Plugin Page 1:  A 'Hello World' SageWorks Plugin Page"""

import dash
from dash import register_page
import dash_bootstrap_components as dbc


class PluginPageExample:
    """Plugin Page:  A SageWorks Plugin Web Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Hello World"

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Register this page with Dash and set up the layout (required)
        register_page(
            __name__,
            path="/",
            name=self.page_name,
            layout=self.page_layout(),
        )

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""

        layout = dash.html.Div(
            children=[
                dbc.Row(
                    [
                        dash.html.H2("SageWorks: Models"),
                        dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                    ]
                ),
            ],
        )
        return layout


if __name__ == "__main__":
    # This class takes in model details and generates a details Markdown component
    from dash import html, dcc, Dash, page_container

    # Create our Dash Application
    app = dash.Dash(
        __name__,
        title="SageWorks Dashboard",
    )
    server = app.server

    # For Multi-Page Applications, we need to create a 'page container' to hold all the pages
    app.layout = html.Div(
        [
            page_container,
        ]
    )

    # Create the Plugin Page and call page_setup
    plugin_page = PluginPageExample()
    plugin_page.page_setup(app)

    # Run the app
    app.run_server(debug=True)