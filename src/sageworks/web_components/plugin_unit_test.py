import dash
from dash import html, Output, Input

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginInputType
from sageworks.api import FeatureSet, Model, Endpoint, Meta
from sageworks.api.pipeline import Pipeline
from sageworks.core.artifacts.graph_core import GraphCore


class PluginUnitTest:
    def __init__(self, plugin_class, input_data=None, **kwargs):
        """A class to unit test a PluginInterface class.

        Args:
            plugin_class (PluginInterface): The PluginInterface class to test
            input_data (Optional): The input data for this plugin (FeatureSet, Model, Endpoint, etc.)
            **kwargs: Additional keyword arguments
        """
        assert issubclass(plugin_class, PluginInterface), "Plugin class must be a subclass of PluginInterface"

        # Get the input type of the plugin
        plugin_input_type = plugin_class.plugin_input_type

        # If the input data is provided, let's store it for when update_properties is called
        self.input_data = input_data
        self.kwargs = kwargs

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

        # Make sure the plugin has a properties attribute (non-empty list of tuples)
        assert hasattr(self.plugin, "properties"), "Plugin must have a 'properties' attribute"
        if not self.plugin.properties:
            raise ValueError("Plugin must have a non-empty 'properties' attribute")

        # Set up the test callback for updating the plugin
        @self.app.callback(
            [Output(component_id, property) for component_id, property in self.plugin.properties],
            [Input("update-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def update_plugin_properties(n_clicks):
            # Simulate updating the plugin with a FeatureSet, Model, Endpoint, or Model Table
            if plugin_input_type == PluginInputType.FEATURE_SET:
                feature_set = self.input_data if self.input_data is not None else FeatureSet("abalone_features")
                updated_properties = self.plugin.update_properties(feature_set, **self.kwargs)
            elif plugin_input_type == PluginInputType.MODEL:
                model = self.input_data if self.input_data is not None else Model("abalone-regression")
                updated_properties = self.plugin.update_properties(
                    model, inference_run="training_holdout", **self.kwargs
                )
            elif plugin_input_type == PluginInputType.ENDPOINT:
                endpoint = self.input_data if self.input_data is not None else Endpoint("abalone-regression-end")
                updated_properties = self.plugin.update_properties(endpoint, **self.kwargs)
            elif plugin_input_type == PluginInputType.PIPELINE:
                pipeline = self.input_data if self.input_data is not None else Pipeline("abalone_pipeline_v1")
                updated_properties = self.plugin.update_properties(pipeline, **self.kwargs)
            elif plugin_input_type == PluginInputType.GRAPH:
                graph = self.input_data if self.input_data is not None else GraphCore("karate_club")
                updated_properties = self.plugin.update_properties(
                    graph, labels="club", hover_text=["club", "degree"], **self.kwargs
                )
            elif plugin_input_type == PluginInputType.MODEL_TABLE:
                model_df = self.input_data if self.input_data is not None else Meta().models()
                updated_properties = self.plugin.update_properties(model_df, **self.kwargs)
            elif plugin_input_type == PluginInputType.PIPELINE_TABLE:
                pipeline_df = self.input_data if self.input_data is not None else Meta().pipelines()
                updated_properties = self.plugin.update_properties(pipeline_df, **self.kwargs)
            else:
                raise ValueError(f"Invalid test type: {plugin_input_type}")

            # Return the updated properties for the plugin
            return updated_properties

        # Set up callbacks for displaying output signals
        for component_id, property in self.plugin.signals:

            @self.app.callback(
                Output(f"test-output-{component_id}-{property}", "children"), Input(component_id, property)
            )
            def display_output_signal(signal_value):
                return f"{signal_value}"

        # Now register any internal callbacks
        self.plugin.register_internal_callbacks()

    def run(self):
        self.app.run_server(debug=True, use_reloader=False)
