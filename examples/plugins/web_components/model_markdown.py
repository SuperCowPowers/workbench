"""A Markdown Plugin Example for details/information about Models"""

# Dash Imports
from dash import html, dcc
import plotly.graph_objects as go

# SageWorks Imports
from sageworks.api import Model
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class MyModelMarkdown(PluginInterface):
    """MyModelMarkdown Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        self.prefix_id = ""
        self.container = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Model Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.prefix_id = component_id
        self.container = html.Div(
            id=self.prefix_id,
            children=[
                html.H3(id=f"{self.prefix_id}-header", children="Model: Loading..."),
                dcc.Markdown(id=f"{self.prefix_id}-details"),
            ],
        )
        return self.container

    def update_contents(self, model: Model, **kwargs):
        """Update the contents for this plugin component.
        Args:
            model (Model): An instantiated Model object
            **kwargs: Additional keyword arguments (unused)
        """
        # Update the html header
        header = f"Model: {model.uuid}"
        self.container.children[0].children = header

        # Get the model summary
        summary = model.summary()
        markdown = ""
        for key, value in summary.items():

            # Chop off the "sageworks_" prefix
            key = key.replace("sageworks_", "")

            # Add to markdown string
            markdown += f"**{key}:** {value}  \n"

        # Update the markdown details
        self.container.children[1].children = markdown


if __name__ == "__main__":
    # This class takes in model details and generates a pie chart
    import dash

    # Test if the Plugin Class is a valid PluginInterface
    assert issubclass(MyModelMarkdown, PluginInterface)

    # Instantiate the MyModelMarkdown class
    my_plugin = MyModelMarkdown()
    my_component = my_plugin.create_component("my_model_markdown")

    # Give the model object to the plugin
    model = Model("abalone-regression")
    my_plugin.update_contents(model)

    # Initialize Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([my_component])
    app.run(debug=True)
