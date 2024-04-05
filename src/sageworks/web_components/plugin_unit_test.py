import dash
from dash import html, Output, Input, callback
from sageworks.web_components.plugin_interface import PluginInterface
from sageworks.api import Model, Endpoint


class PluginUnitTest:
    def __init__(self, plugin_class, test_type="model"):
        """A class to unit test a PluginInterface class.

        Args:
            plugin_class (PluginInterface): The PluginInterface class to test
            test_type (str): The type of test to run (model or endpoint)
        """
        assert issubclass(plugin_class, PluginInterface), "Plugin class must be a subclass of PluginInterface"

        # Instantiate the plugin
        self.plugin = plugin_class()
        self.component = self.plugin.create_component(f"{self.plugin.__class__.__name__.lower()}_test")

        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div([
            self.component,
            html.Button("Update Plugin", id="update-button")  # Button to trigger the callback
        ])

        # Set up the test callback
        @self.app.callback(
            [Output(component_id, property) for component_id, property in self.plugin.slots.items()],
            [Input("update-button", "n_clicks")],
            prevent_initial_call=True
        )
        def update_plugin_contents(n_clicks):
            # Simulate updating the plugin with a new Model
            if test_type == "model":
                model = Model("abalone-regression")
                updated_contents = self.plugin.update_contents(model)
            elif test_type == "endpoint":
                endpoint = Endpoint("abalone-regression-end")
                updated_contents = self.plugin.update_contents(endpoint)
            else:
                raise ValueError("Invalid test type. Must be 'model' or 'endpoint'")

            # Return the updated contents based on the plugin's slots
            return updated_contents

    def run(self):
        self.app.run_server(debug=True)

