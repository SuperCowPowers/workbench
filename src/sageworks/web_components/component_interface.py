"""An abstract class that defines the web component interface for SageWorks"""

from abc import ABC, abstractmethod
from typing import Any, Union
import re
import logging
import plotly.graph_objects as go
from dash import dcc, html, dash_table


class ComponentInterface(ABC):
    """A Stateless Abstract Web Component Interface
    Notes:
      - These methods are ^stateless^, all data should be passed through the
        arguments and the implementations should not reference 'self' variables
      - The 'create_component' method must be implemented by the child class
      - The 'generate_component_figure' is optional (some components don't use Plotly figures)
    """

    log = logging.getLogger("sageworks")

    ComponentTypes = Union[dcc.Graph, dash_table.DataTable, dcc.Markdown, html.Div]
    FigureTypes = Union[go.Figure, str]  # str is used for dcc.Markdown

    @abstractmethod
    def create_component(self, component_id: str, **kwargs: Any) -> ComponentTypes:
        """Create a Dash Component without any data.
        Args:
            component_id (str): The ID of the web component
            kwargs (Any): Any additional arguments to pass to the component
        Returns:
            Union[go.Figure, dcc.Markdown, html.Div]: The Dash Web component
        """
        pass

    def component_id(self) -> str:
        """This helper method returns the component ID for the component
        Returns:
            str: An auto generated component ID
        """

        # Get the plugin class name
        plugin_class_name = self.__class__.__name__

        # Convert the plugin class name to snake case component ID
        component_id = re.sub("([a-z0-9])([A-Z])", r"\1_\2", plugin_class_name).lower()
        return component_id

    @staticmethod
    def message_figure(message: str, figure_height: int = None) -> go.Figure:
        """This helper method creates a waiting figure for the component
        Args:
            message (str): The message to display
            figure_height (int): The height of the figure (default: None)
        Returns:
            go.Figure: A Plotly Figure
        """
        message_figure = go.Figure()
        message_figure.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text=message, showarrow=False, font=dict(size=32)
        )

        layout_options = dict(
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        if figure_height is not None:
            layout_options["height"] = figure_height

        message_figure.update_layout(**layout_options)

        return message_figure
