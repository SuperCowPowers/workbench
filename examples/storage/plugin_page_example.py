"""Plugin Page:  A SageWorks Plugin Web Interface"""

import dash
from dash import register_page, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import logging

# SageWorks Imports
from sageworks.web_components import table
from sageworks.utils.plugin_manager import PluginManager
from sageworks.api.model import Model


class PluginPageExample:
    """Plugin Page:  A SageWorks Plugin Web Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.log = logging.getLogger("sageworks")
        self.app = None
        self.models_table = None
        self.plugin_comps = {}

        # Get our view from the PluginManager
        pm = PluginManager()
        self.my_model_view = pm.get_view("ModelPluginView")

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Save the app for later
        self.app = app

        # Create a table to display the models
        self.models_table = table.Table().create_component(
            "my_model_table", header_color="rgb(60, 60, 60)", row_select="single", max_height=400
        )

        # Load my custom plugin
        pm = PluginManager()
        custom_plugin = pm.get_web_plugin("CustomPlugin")

        # Create a dictionary of plugin components
        component_id = custom_plugin.component_id()
        self.plugin_comps[component_id] = custom_plugin.create_component(component_id)

        # Register this page with Dash and set up the layout (required)
        register_page(
            __name__,
            path="/plugin",
            name="SageWorks - Plugin",
            layout=self.page_layout(),
        )

        # Callback for the model table
        self.model_table_callback()

        # Callbacks for each plugin
        self.plugin_callback(custom_plugin)

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""

        # Generate rows for each plugin
        plugin_rows = [
            dbc.Row(
                plugin,
                style={"padding": "0px 0px 0px 0px"},
            )
            for component_id, plugin in self.plugin_comps.items()
        ]
        layout = dash.html.Div(
            children=[
                dbc.Row(
                    [
                        dash.html.H2("SageWorks: Models"),
                        dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                    ]
                ),
                # A table that lists out all the Models
                dbc.Row(self.models_table),
                # Add the dynamically generated Plugin rows
                *plugin_rows,
            ],
            style={"margin": "30px"},
        )
        return layout

    def model_table_callback(self):
        """Set up the callbacks for the page"""

        @self.app.callback(
            [Output("my_model_table", "columns"), Output("my_model_table", "data")],
            Input("aws-broker-data", "data"),  # View this as an update trigger
        )
        def models_update(serialized_aws_broker_data):
            """Grab our view data and update the table"""
            models = self.my_model_view.view_data()
            models["id"] = range(len(models))
            column_setup_list = table.Table().column_setup(models, markdown_columns=["Model Group"])
            return [column_setup_list, models.to_dict("records")]

    # Updates the plugin component when a row is selected in the model table
    def plugin_callback(self, plugin):
        # Updates the plugin component when a model row is selected
        @self.app.callback(
            Output(plugin.component_id(), "figure"),
            Input("my_model_table", "derived_viewport_selected_row_ids"),
            State("my_model_table", "data"),
            prevent_initial_call=True,
        )
        def update_plugin_figure(selected_rows, table_data):
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                return no_update

            # Get the selected row data and grab the uuid
            selected_row_data = table_data[selected_rows[0]]
            model_uuid = selected_row_data["uuid"]

            # Instantiate the Model and send it to the plugin
            model = Model(model_uuid)
            return plugin.update_properties(model)
