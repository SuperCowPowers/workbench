"""Plugin Page 2:  A 'Hello World' Workbench Plugin Page"""

import dash
from dash import Input, Output, callback, html, dcc, page_container, register_page
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.cached.cached_meta import CachedMeta


class PluginPage2:
    """Plugin Page: A Workbench Plugin Page Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Hello World 2"
        self.models_table = AGTable()
        self.table_component = None
        self.meta = CachedMeta()

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Create a table to display the models
        self.table_component = self.models_table.create_component(
            "my_model_table", header_color="rgb(60, 60, 60)", max_height=400
        )

        # Register this page with Dash and set up the layout
        register_page(
            __file__,
            path="/plugin_2",
            name=self.page_name,
            layout=self.page_layout(),
        )

        # Set up the callbacks for the page
        self.page_callbacks()

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        layout = dash.html.Div(
            children=[
                dash.html.H1(self.page_name),
                dbc.Row(self.table_component),
                # Interval that triggers once on page load
                dcc.Interval(id="plugin-2-page-load", interval=100, max_intervals=1),
            ]
        )
        return layout

    def page_callbacks(self):
        """Set up the callbacks for the page"""

        @callback(
            [Output(component_id, prop) for component_id, prop in self.models_table.properties],
            [Input("plugin-2-page-load", "n_intervals")],
        )
        def _populate_models_table(_n_intervals):
            """Callback to Populate the models table with data"""
            models = self.meta.models(details=True)
            models["uuid"] = models["Model Group"]
            return self.models_table.update_properties(models)


# Unit Test for your Plugin Page
if __name__ == "__main__":
    import webbrowser
    from workbench.utils.theme_manager import ThemeManager

    # Set up the Theme Manager
    tm = ThemeManager()
    tm.set_theme("quartz_dark")
    css_files = tm.css_files()

    # Create the Dash app
    my_app = dash.Dash(
        __name__, title="Workbench Dashboard", use_pages=True, external_stylesheets=css_files, pages_folder=""
    )
    my_app.layout = html.Div(
        [
            dbc.Container([page_container], fluid=True, className="dbc dbc-ag-grid"),
        ],
        **{"data-bs-theme": tm.data_bs_theme()},
    )

    # Create the Plugin Page and call page_setup
    plugin_page = PluginPage2()
    plugin_page.page_setup(my_app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/plugin_2")

    # Note: This 'main' is purely for running/testing locally
    my_app.run(host="localhost", port=8000, debug=True)
