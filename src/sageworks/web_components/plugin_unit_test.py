import dash
from dash import html, Output, Input

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginInputType
from sageworks.api import Model, Endpoint, Meta
from sageworks.api.pipeline import Pipeline


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

        # Set up the layout
        layout_children = [self.component, html.Button("Update Plugin", id="update-button")]

        # Signal output displays
        layout_children.append(html.H3("Signals:"))
        for component_id, property in self.plugin.signals:
            # A Row with the component ID and property and an output div
            layout_children.append(html.H4(f"Property: {property}"))
            layout_children.append(html.Div(id=f"test-output-{component_id}-{property}"))

        self.app.layout = html.Div(layout_children)

        # Set up the test callback for updating the plugin
        @self.app.callback(
            [Output(component_id, property) for component_id, property in self.plugin.properties],
            [Input("update-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def update_plugin_properties(n_clicks):
            # Simulate updating the plugin with a new Model, Endpoint, or Model Table
            if plugin_input_type == PluginInputType.MODEL:
                model = Model("abalone-regression")
                updated_proporties = self.plugin.update_properties(model, inference_run="training_holdout")
            elif plugin_input_type == PluginInputType.ENDPOINT:
                endpoint = Endpoint("abalone-regression-end")
                updated_proporties = self.plugin.update_properties(endpoint)
            elif plugin_input_type == PluginInputType.PIPELINE:
                pipeline = Pipeline("abalone_pipeline_v1")
                updated_proporties = self.plugin.update_properties(pipeline)
            elif plugin_input_type == PluginInputType.MODEL_TABLE:
                model_df = Meta().models()
                updated_proporties = self.plugin.update_properties(model_df)
            elif plugin_input_type == PluginInputType.PIPELINE_TABLE:
                pipeline_df = Meta().pipelines()
                updated_proporties = self.plugin.update_properties(pipeline_df)
            else:
                raise ValueError(f"Invalid test type: {plugin_input_type}")

            # Return the updated properties for the plugin
            return updated_proporties

        # Set up callbacks for displaying output signals
        for component_id, property in self.plugin.signals:

            @self.app.callback(
                Output(f"test-output-{component_id}-{property}", "children"), Input(component_id, property)
            )
            def display_output_signal(signal_value):
                return f"{signal_value}"

    def run(self):
        self.app.run_server(debug=True)
