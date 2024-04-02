"""Plugin Page 1:  A 'Hello World' SageWorks Plugin Page"""

import dash
from dash import html, page_container, register_page, callback, Output, Input
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.web_components import table
from sageworks.api.meta import Meta


class PluginPage2:
    """Plugin Page:  A SageWorks Plugin Web Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Hello World"
        self.models_table = table.Table()
        self.table_component = None
        self.meta = Meta()

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Create a table to display the models
        self.table_component = self.models_table.create_component(
            "my_model_table", header_color="rgb(60, 60, 60)", row_select="single", max_height=400
        )

        # Register this page with Dash and set up the layout (required)
        register_page(
            __name__,
            path="/plugin_2",
            name=self.page_name,
            layout=self.page_layout(),
        )

        # Populate the models table with data
        models = self.meta.models()
        models["uuid"] = models["Model Group"]
        models["id"] = range(len(models))
        self.table_component.columns = self.models_table.column_setup(models)
        self.table_component.data = models.to_dict("records")

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        layout = dash.html.Div(
            children=[
                dash.html.H1(self.page_name),
                dbc.Row(self.table_component),
            ]
        )
        return layout


# Unit Test for your Plugin Page
if __name__ == "__main__":
    import webbrowser

    # Create our Dash Application
    app = dash.Dash(
        __name__,
        title="SageWorks Dashboard",
        use_pages=True,
        pages_folder="",
        external_stylesheets=[dbc.themes.DARKLY],
    )

    # For Multi-Page Applications, we need to create a 'page container' to hold all the pages
    app.layout = html.Div([page_container])

    # Create the Plugin Page and call page_setup
    plugin_page = PluginPage2()
    plugin_page.page_setup(app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/plugin_2")

    # Note: This 'main' is purely for running/testing locally
    app.run(host="0.0.0.0", port=8000, debug=True)
