"""Plugin Page 3:  A 'Hello World' Workbench Plugin Page"""

import dash
from dash import html, dcc, page_container, register_page, callback, Output, Input, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.web_interface.components.plugins.model_details import ModelDetails
from workbench.web_interface.components.model_plot import ModelPlot
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_model import CachedModel


class PluginPage3:
    """Plugin Page:  A Workbench Plugin Page Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Hello World 3"
        self.models_table = AGTable()
        self.table_component = None
        self.model_details = ModelDetails()
        self.details_component = None
        self.model_plot = ModelPlot()
        self.plot_component = None
        self.meta = CachedMeta()
        self.plugins = [self.model_details]  # Add any additional plugins here

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Create a table to display the models
        self.table_component = self.models_table.create_component(
            "plugin_3_model_table", header_color="rgb(60, 60, 60)", max_height=400
        )

        # Create a model details panel and model plot
        self.details_component = self.model_details.create_component("plugin_3_model_details")
        self.plot_component = self.model_plot.create_component("plugin_3_model_plot")

        # Register this page with Dash and set up the layout
        register_page(
            __file__,
            path="/plugin_3",
            name=self.page_name,
            layout=self.page_layout(),
        )

        # Load page callbacks
        self.page_load_callbacks()

        # Register the callbacks for the page
        self.page_callbacks()
        self.setup_plugin_callbacks()
        self.model_details.register_internal_callbacks()

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        layout = dash.html.Div(
            children=[
                dash.html.H1(self.page_name),
                dbc.Row(self.table_component),
                dbc.Row(
                    [
                        dbc.Col(self.details_component, width=5),
                        dbc.Col(self.plot_component, width=7),
                    ]
                ),
                # Interval that triggers once on page load
                dcc.Interval(id="plugin-3-page-load", interval=100, max_intervals=1),
            ]
        )
        return layout

    def page_load_callbacks(self):
        """Load page (once) callbacks"""

        @callback(
            [Output(component_id, prop) for component_id, prop in self.models_table.properties],
            [Input("plugin-3-page-load", "n_intervals")],
        )
        def _populate_models_table(_n_intervals):
            """Callback to Populate the models table with data"""
            models = self.meta.models(details=True)
            models["uuid"] = models["Model Group"]
            return self.models_table.update_properties(models)

    def page_callbacks(self):
        """Register the callbacks for the page"""

        @callback(
            Output("plugin_3_model_plot", "figure"),
            Input("plugin_3_model_details-dropdown", "value"),
            Input("plugin_3_model_table", "selectedRows"),
            prevent_initial_call=True,
        )
        def generate_model_plot_figure(inference_run, selected_rows):
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                return no_update

            # Get the selected row data and grab the uuid
            selected_row_data = selected_rows[0]
            model_uuid = selected_row_data["uuid"]
            m = CachedModel(model_uuid)

            # Model Details Markdown component
            model_plot_fig = self.model_plot.update_properties(m, inference_run)

            # Return the details/markdown for these data details
            return model_plot_fig

    def setup_plugin_callbacks(self):
        @callback(
            # Aggregate plugin outputs
            [Output(component_id, prop) for p in self.plugins for component_id, prop in p.properties],
            Input("plugin_3_model_details-dropdown", "value"),
            Input("plugin_3_model_table", "selectedRows"),
        )
        def update_all_plugin_properties(inference_run, selected_rows):
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                raise PreventUpdate

            # Get the selected row data and grab the uuid
            selected_row_data = selected_rows[0]
            object_uuid = selected_row_data["uuid"]

            # Create the Model object
            model = CachedModel(object_uuid)

            # Update all the properties for each plugin
            all_props = []
            for p in self.plugins:
                all_props.extend(p.update_properties(model, inference_run=inference_run))

            # Return all the updated properties
            return all_props


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
    plugin_page = PluginPage3()
    plugin_page.page_setup(my_app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/plugin_3")

    # Note: This 'main' is purely for running/testing locally
    my_app.run(host="localhost", port=8000, debug=True)
