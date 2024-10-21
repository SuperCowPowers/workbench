"""An abstract class that defines the web component interface for SageWorks"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Union
import traceback
import re
from functools import wraps
import plotly.graph_objects as go
import pandas as pd
from dash import dcc
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
import networkx as nx

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet, Model, Endpoint
from sageworks.api.pipeline import Pipeline
from sageworks.core.artifacts.graph_core import GraphCore

log = logging.getLogger("sageworks")


def get_reversed_stack_trace(exception, limit=5):
    # Get the full stack trace
    full_stack = traceback.format_exception(type(exception), exception, exception.__traceback__)
    # Reverse the stack and take the last `limit` lines
    relevant_stack = full_stack[-limit:]
    return "".join(relevant_stack)


class ComponentInterface(ABC):
    """A Abstract Web Component Interface
    Notes:
      - The 'create_container' method create a gcc.Graph, html.Div, etc
      - The 'update_properties' method generates the property values
    """

    log = logging.getLogger("sageworks")

    SageworksObject = Union[DataSource, FeatureSet, Model, Endpoint, Pipeline, GraphCore, nx.Graph, pd.DataFrame]

    def __init__(self, theme: str = "DARK"):
        """Initialize the Component Interface

        Args:
            theme (str): The theme to use for the component ("LIGHT" or "DARK" default is "DARK")
        """
        self.component_id = None
        self.container = None
        self.properties = []
        self.signals = []

        # Store the theme for the Web Component
        if theme == "DARK":
            self.theme = dbc.themes.DARKLY
            self.dark_theme = True
        else:
            self.theme = dbc.themes.BOOTSTRAP
            self.dark_theme = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Automatically apply the error handling decorator to the create_component and update_properties methods
        if hasattr(cls, "create_component") and callable(cls.create_component):
            cls.create_component = create_component_handler(cls.create_component)
        if hasattr(cls, "update_properties") and callable(cls.update_properties):
            cls.update_properties = update_properties_handler(cls.update_properties)

    @abstractmethod
    def create_component(self, component_id: str, **kwargs: Any) -> Component:
        """Create a Dash Component/Container without any data.

        Args:
            component_id (str): The ID of the web component
            kwargs (Any): Any additional arguments to pass to the component

        Returns:
           Component: A Dash Base Component
        """
        pass

    def update_properties(self, data_object: SageworksObject) -> list:
        """Update the properties of the component/container

        Args:
            data_object (sageworks_object/dataframe): A SageWorks object or DataFrame

        Returns:
            list: A list of the updated property values for the component
        """
        pass

    def generate_component_id(self) -> str:
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
    def display_text(text_message: str, figure_height: int = None, font_size=32) -> go.Figure:
        """This helper method displays a text message figure for the component
        Args:
            text_message (str): The text message to display
            figure_height (int): The height of the figure (default: None)
            font_size (int): The font size of the message (default: 32)
        Returns:
            go.Figure: A Plotly Figure
        """
        text_display_text = go.Figure()

        # If the text message has any \n characters, replace them with <br> for HTML
        text_message = text_message.replace("\n", "<br>")

        text_display_text.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=text_message,
            showarrow=False,
            font=dict(size=font_size),
        )

        layout_options = dict(
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        if figure_height is not None:
            layout_options["height"] = figure_height

        text_display_text.update_layout(**layout_options)

        return text_display_text


# These are helper decorators to catch errors in plugin methods
def create_component_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the class name of the plugin
            class_name = args[0].__class__.__name__ if args else "UnknownPlugin"

            # Generate our stack trace figure
            figure, _ = stack_trace_figure(class_name, e)
            return dcc.Graph(id="error", figure=figure)

    return wrapper


def update_properties_handler(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            # Get the class name of the plugin
            class_name = self.__class__.__name__

            # Generate our stack trace figure
            figure, error_info = stack_trace_figure(class_name, e)

            # Prepare the error output to match the properties format
            error_output = []
            for component_id, property in self.properties:
                if property == "figure":
                    error_output.append(figure)
                elif property in ["children", "value", "data"]:
                    error_output.append(error_info)
                elif property == "columnDefs":
                    error_output.append([{"headerName": "Crash", "field": "Crash"}])
                elif property == "rowData":
                    error_output.append([{"Crash": error_info}])
                else:
                    error_output.append(None)  # Fallback for other properties
            return error_output

    return wrapper


def stack_trace_figure(class_name: str, e: Exception) -> tuple[go.Figure, str]:
    """This helper method returns a Plotly Figure and the UI error info with the stack trace of an exception
    Args:
        class_name (str): The class name of the plugin
        e (Exception): The exception object
    Returns:
        tuple: A tuple containing the Plotly Figure and the stack trace error info
    """

    # Get the reversed stack trace
    reversed_stack_trace = get_reversed_stack_trace(e)
    header = f"{class_name} Crashed"
    stack_output = f"{header}: \n{reversed_stack_trace}"
    log.critical(stack_output)

    # Now split lines that are too long
    max_len = 80
    wrapped_stack_output = "\n".join(
        [line[i : i + max_len] for line in stack_output.split("\n") for i in range(0, len(line), max_len)]
    )

    # Send the wrapped stack output as a figure with text
    figure = ComponentInterface.display_text(wrapped_stack_output, figure_height=400, font_size=16)

    return figure, stack_output  # Return both figure and the stack trace
