"""An Example Model plugin component"""

from dash import dcc
import random
import plotly.graph_objects as go

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.model import Model


class ModelPlugin(PluginInterface):
    """MyModelPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        self.container = None

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Model Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.container = dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))
        return self.container

    def update_contents(self, model: Model, **kwargs):
        """Create a Figure for the plugin.
        Args:
            model (Model): An instantiated Model object
            **kwargs: Additional keyword arguments (unused)
        """
        model_name = f"Model: {model.uuid}"

        # Generate random values for the pie chart
        pie_values = [random.randint(10, 30) for _ in range(3)]

        # Create a pie chart with the endpoint name as the title
        fig = go.Figure(data=[go.Pie(labels=["A", "B", "C"], values=pie_values)], layout=go.Layout(title=model_name))
        self.container.figure = fig


if __name__ == "__main__":
    # This class takes in model details and generates a pie chart
    from dash import html, Dash

    # Test if the Plugin Class is a valid PluginInterface
    assert issubclass(ModelPlugin, PluginInterface)

    # Instantiate the EndpointTurbo class
    my_plugin = ModelPlugin()
    my_container = my_plugin.create_component("model_plugin")

    # Give the model object to the plugin
    model = Model("abalone-regression")
    my_plugin.update_contents(model)

    # Initialize Dash app
    app = Dash(__name__)

    app.layout = html.Div([my_container])
    app.run(debug=True)
