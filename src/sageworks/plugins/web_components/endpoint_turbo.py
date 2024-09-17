"""A EndpointTurbo plugin component"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.endpoint import Endpoint


class EndpointTurbo(PluginInterface):
    """EndpointTurbo Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.ENDPOINT
    plugin_input_type = PluginInputType.ENDPOINT

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a EndpointTurbo Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.component_id = component_id
        self.container = dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

        # Fill in plugin properties
        self.properties = [(self.component_id, "figure")]

        # Return the container
        return self.container

    def update_properties(self, endpoint: Endpoint, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            endpoint (Endpoint): An instantiated Model object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """

        data = [  # Portfolio (inner donut)
            # Inner ring
            go.Pie(
                values=[20, 40],
                labels=["Reds", "Blues"],
                domain={"x": [0.05, 0.45], "y": [0.2, 0.8]},
                hole=0.5,
                direction="clockwise",
                sort=False,
                marker={"colors": ["#CB4335", "#2E86C1"]},
            ),
            # Outer ring
            go.Pie(
                values=[5, 15, 30, 10],
                labels=["Medium Red", "Light Red", "Medium Blue", "Light Blue"],
                domain={"x": [0.05, 0.45], "y": [0, 1]},
                hole=0.75,
                direction="clockwise",
                sort=False,
                marker={"colors": ["#EC7063", "#F1948A", "#5DADE2", "#85C1E9"]},
                showlegend=False,
            ),
            # Inner ring
            go.Pie(
                values=[20, 40],
                labels=["Greens", "Oranges"],
                domain={"x": [0.55, 0.95], "y": [0.2, 0.8]},
                hole=0.5,
                direction="clockwise",
                sort=False,
                marker={"colors": ["#558855", "#DD9000"]},
            ),
            # Outer ring
            go.Pie(
                values=[5, 15, 30, 10],
                labels=["Medium Green", "Light Green", "Medium Orange", "Light Orange"],
                domain={"x": [0.55, 0.95], "y": [0, 1]},
                hole=0.75,
                direction="clockwise",
                sort=False,
                marker={"colors": ["#668866", "#779977", "#EEA540", "#FFC060"]},
                showlegend=False,
            ),
        ]

        # Create the nested pie chart plot with custom settings
        endpoint_name = f"Endpoint: {endpoint.uuid}"
        turbo_figure = go.Figure(data=data)
        turbo_figure.update_layout(
            margin={"t": 30, "b": 10, "r": 10, "l": 10, "pad": 10}, title=endpoint_name, height=400
        )

        # Return the updated property values
        return [turbo_figure]


if __name__ == "__main__":
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(EndpointTurbo).run()
