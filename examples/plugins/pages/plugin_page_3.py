"""Plugin Page 1:  A 'Hello World' SageWorks Plugin Page"""

import dash
from dash import html, page_container, register_page, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.web_components import table
from sageworks.api.meta import Meta
from sageworks.web_components.model_details import ModelDetails
from sageworks.web_components.model_plot import ModelPlot
from sageworks.api.model import Model


class PluginPage3:
    """Plugin Page:  A SageWorks Plugin Web Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Hello World"
        self.models_table = table.Table()
        self.table_component = None
        self.model_details = ModelDetails()
        self.details_component = None
        self.model_plot = ModelPlot()
        self.plot_component = None
        self.meta = Meta()

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Create a table to display the models
        self.table_component = self.models_table.create_component(
            "plugin_3_model_table", header_color="rgb(60, 60, 60)", row_select="single", max_height=400
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

        # Populate the models table with data
        models = self.meta.models()
        models["uuid"] = models["Model Group"]
        models["id"] = range(len(models))
        self.table_component.columns = self.models_table.column_setup(models)
        self.table_component.data = models.to_dict("records")

        # Register the callbacks
        self.register_app_callbacks(app)
        self.model_details.register_callbacks("plugin_3_model_table")

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
            ]
        )
        return layout

    def register_app_callbacks(self, app: dash.Dash):
        """Register the callbacks for the page"""

        @app.callback(
            Output("plugin_3_model_plot", "figure"),
            [
                Input("plugin_3_model_details-dropdown", "value"),
                Input("plugin_3_model_table", "derived_viewport_selected_row_ids"),
            ],
            State("plugin_3_model_table", "data"),
            prevent_initial_call=True,
        )
        def generate_model_plot_figure(inference_run, selected_rows, table_data):
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                return no_update

            # Get the selected row data and grab the uuid
            selected_row_data = table_data[selected_rows[0]]
            model_uuid = selected_row_data["uuid"]
            model = Model(model_uuid, legacy=True)

            # Model Details Markdown component
            model_plot_fig = self.model_plot.update_properties(model, inference_run)

            # Return the details/markdown for these data details
            return model_plot_fig


# Unit Test for your Plugin Page
if __name__ == "__main__":
    import webbrowser

    # Create our Dash Application
    my_app = dash.Dash(
        __name__, title="SageWorks Dashboard", use_pages=True, pages_folder="", external_stylesheets=[dbc.themes.DARKLY]
    )

    # For Multi-Page Applications, we need to create a 'page container' to hold all the pages
    my_app.layout = html.Div([page_container])

    # Create the Plugin Page and call page_setup
    plugin_page = PluginPage3()
    plugin_page.page_setup(my_app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/plugin_3")

    # Note: This 'main' is purely for running/testing locally
    my_app.run(host="localhost", port=8000, debug=True)
