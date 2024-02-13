"""A EndpointTurbo plugin component"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginType, PluginInputType
from sageworks.api.endpoint import Endpoint


class EndpointTurbo(PluginInterface):
    """EndpointTurbo Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.ENDPOINT
    plugin_input_type = PluginInputType.ENDPOINT_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a EndpointTurbo Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        return dcc.Graph(id=component_id, figure=self.message_figure("Waiting for Data..."))

    def generate_component_figure(self, endpoint: Endpoint) -> go.Figure:
        """Create a EndpointTurbo Figure for the numeric columns in the dataframe.
        Args:
            endpoint (Endpoint): An instantiated Endpoint object
        Returns:
            go.Figure: A Plotly Figure object
        """

        # Just to make sure we have the right endpoint object
        print(endpoint.summary())

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
        fig = go.Figure(data=data)
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10}, height=400)

        return fig


if __name__ == "__main__":
    # This class takes in model details and generates a EndpointTurbo
    from sageworks.api.endpoint import Endpoint

    # Instantiate an Endpoint
    end = Endpoint("abalone-regression-end")

    # Instantiate the EndpointTurbo class
    pie = EndpointTurbo()

    # Generate the figure
    fig = pie.generate_component_figure(end)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
