"""An abstract class that defines the web component interface for SageWorks"""

from abc import ABC, abstractmethod
from typing import Any, Union
import re
from functools import wraps
import logging
import plotly.graph_objects as go
from dash import dcc, html, dash_table

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet, Model, Endpoint


class ComponentInterface(ABC):
    """A Stateless Abstract Web Component Interface
    Notes:
      - These methods are ^stateless^, all data should be passed through the
        arguments and the implementations should not reference 'self' variables
      - The 'create_component' method must be implemented by the child class
      - The 'generate_figure' is optional (some components don't use Plotly figures)
    """

    log = logging.getLogger("sageworks")

    SageworksObject = Union[DataSource, FeatureSet, Model, Endpoint]
    ComponentTypes = Union[dcc.Graph, dash_table.DataTable, dcc.Markdown, html.Div]
    FigureTypes = Union[go.Figure, str]  # str is used for dcc.Markdown

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Automatically apply the error handling decorator to the create_component and generate_figure methods
        if hasattr(cls, "create_component") and callable(cls.create_component):
            cls.create_component = component_error_decorator(cls.create_component)
        if hasattr(cls, "generate_figure") and callable(cls.generate_figure):
            cls.generate_figure = figure_error_decorator(cls.generate_figure)

    @abstractmethod
    def create_component(self, component_id: str, **kwargs: Any) -> ComponentTypes:
        """Create a Dash Component without any data.
        Args:
            component_id (str): The ID of the web component
            kwargs (Any): Any additional arguments to pass to the component
        Returns:
            Union[dcc.Graph, dash_table.DataTable, dcc.Markdown, html.Div]: The Dash Web component
        """
        pass

    def generate_figure(self, data_object: SageworksObject) -> FigureTypes:
        """Generate a figure from the data in the given dataframe.
        Args:
            data_object (sageworks_object): The instantiated data object for the plugin type.
        Returns:
            Union[go.Figure, str]: A Plotly Figure or a Markdown string
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
    def message_figure(text_message: str, figure_height: int = None, font_size=32) -> go.Figure:
        """This helper method creates a waiting figure for the component
        Args:
            text_message (str): The text message to display
            figure_height (int): The height of the figure (default: None)
            font_size (int): The font size of the message (default: 32)
        Returns:
            go.Figure: A Plotly Figure
        """
        text_message_figure = go.Figure()
        text_message_figure.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text=text_message, showarrow=False, font=dict(size=font_size)
        )

        layout_options = dict(
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        if figure_height is not None:
            layout_options["height"] = figure_height

        text_message_figure.update_layout(**layout_options)

        return text_message_figure


# These are helper decorators to catch errors in plugin methods
def component_error_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the class name of the plugin
            class_name = args[0].__class__.__name__ if args else "UnknownPlugin"
            error_info = f"{class_name} Crashed: {e.__class__.__name__}: {e}"
            figure = ComponentInterface.message_figure(error_info, figure_height=100, font_size=16)
            return dcc.Graph(id="error", figure=figure)

    return wrapper


def figure_error_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the class name of the plugin
            class_name = args[0].__class__.__name__ if args else "UnknownPlugin"
            error_info = f"{class_name} Crashed: {e.__class__.__name__}: {e}"
            return ComponentInterface.message_figure(error_info, figure_height=100, font_size=16)

    return wrapper
