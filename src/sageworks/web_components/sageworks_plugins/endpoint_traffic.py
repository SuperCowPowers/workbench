"""A line chart component"""
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginType, PluginInputType


class EndpointTraffic(PluginInterface):
    """Endpoint Traffic Component"""

    plugin_type = PluginType.ENDPOINT
    plugin_input_type = PluginInputType.ENDPOINT_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Endpoint Traffic Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Endpoint Traffic Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, endpoint_details: dict) -> go.Figure:
        df = px.data.stocks(indexed=True) - 1
        df.columns.name = "Endpoints"
        # TEMP
        df.rename(
            {
                "Date": "awesome",
                "GOOG": "aqsol-regression 1",
                "AAPL": "aqsol-regression 2",
                "AMZN": "abalone-regression 1",
                "FB": "abalone-regression 2",
                "NFLX": "super-secret 1",
                "MSFT": "super-secret 2",
            },
            axis=1,
            inplace=True,
        )
        return px.area(df, facet_col="Endpoints", facet_col_wrap=2)
