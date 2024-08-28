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
import networkx as nx

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet, Model, Endpoint
from sageworks.api.pipeline import Pipeline
from sageworks.core.artifacts.graph_core import GraphCore

log = logging.getLogger("sageworks")


def get_reversed_stack_trace(exception, limit=2):
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

    def __init__(self):
        self.component_id = None
        self.container = None
        self.properties = []
        self.signals = []

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
            # Get a few lines from the stack trace
            stack_trace = "".join(traceback.format_exception(e))
            header = f"{class_name} Crashed"
            log.critical(f"{header}: {stack_trace}")
            ui_error_info = f"{header}: {get_reversed_stack_trace(e)}"
            figure = ComponentInterface.display_text(ui_error_info, figure_height=400, font_size=14)
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
            # Get a few lines from the stack trace
            stack_trace = "".join(traceback.format_exception(e))
            header = f"{class_name} Crashed"
            log.critical(f"{header}: {stack_trace}")
            ui_error_info = f"{header}: {get_reversed_stack_trace(e)}"
            figure = ComponentInterface.display_text(ui_error_info, figure_height=400, font_size=14)

            # Prepare the error output to match the properties format
            error_output = []
            for component_id, property in self.properties:
                if property == "figure":
                    error_output.append(figure)
                elif property in ["children", "value", "data"]:
                    error_output.append(ui_error_info)
                elif property == "columnDefs":
                    error_output.append([{"headerName": "Crash", "field": "Crash"}])
                elif property == "rowData":
                    error_output.append([{"Crash": ui_error_info}])
                else:
                    error_output.append(None)  # Fallback for other properties
            return error_output

    return wrapper
