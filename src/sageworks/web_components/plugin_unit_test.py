import dash
from dash import html, Output, Input
from sageworks.web_components.plugin_interface import PluginInterface, PluginInputType
from sageworks.api import Model, Endpoint, Meta


class PluginUnitTest:
    def __init__(self, plugin_class):
        """A class to unit test a PluginInterface class.

        Args:
            plugin_class (PluginInterface): The PluginInterface class to test
        """
        assert issubclass(plugin_class, PluginInterface), "Plugin class must be a subclass of PluginInterface"

        # Get the input type of the plugin
        plugin_input_type = plugin_class.plugin_input_type

        # Instantiate the plugin
        self.plugin = plugin_class()
        self.component = self.plugin.create_component(f"{self.plugin.__class__.__name__.lower()}_test")

        # Create the Dash app
        self.app = dash.Dash(__name__)

        # Setup the layout
        self.app.layout = html.Div(
            [self.component, html.Button("Update Plugin", id="update-button")]  # Button to trigger the callback
        )

        # Set up the test callback
        @self.app.callback(
            [Output(component_id, property) for component_id, property in self.plugin.content_slots],
            [Input("update-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def update_plugin_contents(n_clicks):
            # Simulate updating the plugin with a new Model
            if plugin_input_type == PluginInputType.MODEL:
                model = Model("abalone-regression")
                updated_contents = self.plugin.update_contents(model)
            elif plugin_input_type == PluginInputType.ENDPOINT:
                endpoint = Endpoint("abalone-regression-end")
                updated_contents = self.plugin.update_contents(endpoint)
            elif plugin_input_type == PluginInputType.MODEL_TABLE:
                model_table = Meta().models()
                updated_contents = self.plugin.update_contents(model_table)
            else:
                raise ValueError(f"Invalid test type: {plugin_input_type}")

            # Return the updated contents based on the plugin's slots
            return updated_contents

    def run(self):
        self.app.run_server(debug=True)
