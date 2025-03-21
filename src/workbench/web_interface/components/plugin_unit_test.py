from dash import html, Dash, Output, Input
import dash_bootstrap_components as dbc
import logging
import socket


# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginInputType
from workbench.api import DataSource, FeatureSet, Model, Endpoint, Meta
from workbench.api.compound import Compound
from workbench.api.graph_store import GraphStore
from workbench.api.pipeline import Pipeline
from workbench.utils.theme_manager import ThemeManager

# Setup Logging
log = logging.getLogger("workbench")


class PluginUnitTest:
    def __init__(self, plugin_class, theme="dark", input_data=None, auto_update=True, **kwargs):
        """A class to unit test a PluginInterface class.

        Args:
            plugin_class (PluginInterface): The PluginInterface class to test
            theme (str): The theme to use for the Dash app (default: "dark")
            input_data (Optional): The input data for this plugin (FeatureSet, Model, Endpoint, or DataFrame)
            auto_update (bool): Whether to automatically update the plugin properties (default: True)
            **kwargs: Additional keyword arguments
        """
        assert issubclass(
            plugin_class, PluginInterface
        ), "Plugin class has not passed all the PluginInterface validations"

        # If the input data is provided, let's store it for when update_properties is called
        self.input_data = input_data
        self.kwargs = kwargs

        # Set up the Theme Manager
        tm = ThemeManager()
        tm.set_theme(theme)
        css_files = tm.css_files()

        # Create the Dash app with the theme CSS files
        self.app = Dash(
            __name__,
            title="Plugin Unit Test",
            external_stylesheets=css_files,
        )
        self.port = 8050

        # Register the CSS route in the ThemeManager
        tm.register_css_route(self.app)

        # Instantiate the plugin
        self.plugin = plugin_class()
        self.component = self.plugin.create_component(f"{self.plugin.__class__.__name__.lower()}_test")

        # Set up the layout
        layout_children = [
            dbc.Row(self.component, style={"marginBottom": "20px", "height": "700px"}),
            html.Button("Update Plugin", id="update-button", style={"marginBottom": "20px"}),
        ]

        # Signal output displays
        layout_children.append(html.H3("Signals:"))
        for component_id, property in self.plugin.signals:
            # A Row with the component ID and property and an output div
            layout_children.append(html.H4(f"Property: {property}"))
            layout_children.append(html.Div(id=f"test-output-{component_id}-{property}"))

        # Set the layout
        self.app.layout = html.Div(
            [
                dbc.Container(layout_children, fluid=True, className="dbc dbc-ag-grid"),
            ],
            style={"margin": "40px"},
            **{"data-bs-theme": tm.data_bs_theme()},
        )

        # Make sure the plugin has a properties attribute (non-empty list of tuples)
        assert hasattr(self.plugin, "properties"), "Plugin must have a 'properties' attribute"
        if not self.plugin.properties:
            raise ValueError("Plugin must have a non-empty 'properties' attribute")

        # Call the internal method to update properties
        if auto_update:
            self._trigger_update()

        # Set up the test callback for updating the plugin
        @self.app.callback(
            [Output(component_id, property) for component_id, property in self.plugin.properties],
            [Input("update-button", "n_clicks")],
        )
        def update_plugin_properties(n_clicks):
            print("Update Plugin Properties")
            return self._trigger_update()

        # Set up callbacks for displaying output signals
        for component_id, property in self.plugin.signals:

            @self.app.callback(
                Output(f"test-output-{component_id}-{property}", "children"), Input(component_id, property)
            )
            def display_output_signal(signal_value):
                return f"{signal_value}"

        # Now register any internal callbacks
        self.plugin.register_internal_callbacks()

    def _trigger_update(self):
        """Trigger an update for the plugin properties based on its input type."""
        plugin_input_type = self.plugin.plugin_input_type

        if plugin_input_type == PluginInputType.DATA_SOURCE:
            data_source = self.input_data if self.input_data is not None else DataSource("abalone_data")
            return self.plugin.update_properties(data_source, **self.kwargs)
        elif plugin_input_type == PluginInputType.FEATURE_SET:
            feature_set = self.input_data if self.input_data is not None else FeatureSet("abalone_features")
            return self.plugin.update_properties(feature_set, **self.kwargs)
        elif plugin_input_type == PluginInputType.MODEL:
            model = self.input_data if self.input_data is not None else Model("abalone-regression")
            return self.plugin.update_properties(model, inference_run="auto_inference", **self.kwargs)
        elif plugin_input_type == PluginInputType.ENDPOINT:
            endpoint = self.input_data if self.input_data is not None else Endpoint("abalone-regression")
            return self.plugin.update_properties(endpoint, **self.kwargs)
        elif plugin_input_type == PluginInputType.PIPELINE:
            pipeline = self.input_data if self.input_data is not None else Pipeline("abalone_pipeline_v1")
            return self.plugin.update_properties(pipeline, **self.kwargs)
        elif plugin_input_type == PluginInputType.GRAPH:
            graph = self.input_data if self.input_data is not None else GraphStore().get("test/karate_club")
            return self.plugin.update_properties(graph, **self.kwargs)
        elif plugin_input_type == PluginInputType.DATAFRAME:
            df = self.input_data if self.input_data is not None else Meta().models(details=True)
            return self.plugin.update_properties(df, **self.kwargs)
        elif plugin_input_type == PluginInputType.COMPOUND:
            fake_compound = Compound("AQSOL-0001")
            fake_compound.smiles = "CC(C)C1=CC=C(C=C1)C(=O)O"
            compound = self.input_data if self.input_data is not None else fake_compound
            return self.plugin.update_properties(compound, **self.kwargs)
        else:
            raise ValueError(f"Invalid test type: {plugin_input_type}")

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
        """Run the Dash server for the plugin, handling common errors gracefully."""
        while self.is_port_in_use(self.port):
            log.info(f"Port {self.port} is in use. Trying the next one...")
            self.port += 1  # Increment the port number until an available one is found

        log.info(f"Starting Dash server on port {self.port}...")
        self.app.run_server(debug=True, use_reloader=False, port=self.port)

    @staticmethod
    def is_port_in_use(port):
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(("127.0.0.1", port)) == 0
