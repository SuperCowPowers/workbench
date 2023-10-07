"""An abstract class that defines the web component interface for SageWorks"""
from abc import ABC, abstractmethod
from typing import Any, Union
import re
import plotly.graph_objects as go
import pandas as pd
from dash import dcc, html, dash_table


class ComponentInterface(ABC):
    """A Stateless Abstract Web Component Interface
    Notes:
      - These methods are ^stateless^, all data should be passed through the
        arguments and the implementations should not reference 'self' variables
      - The 'create_component' method must be implemented by the child class
      - The 'generate_component_figure' is optional (some components don't use Plotly figures)
    """

    ComponentTypes = Union[go.Figure, dash_table.DataTable, dcc.Markdown, html.Div]

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

    def generate_component_figure(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate a figure from the data in the given dataframe.
        Args:
            df (pd.DataFrame): The dataframe to generate the figure from.
            kwargs (Any): Any additional arguments to pass for figure generation.
        Returns:
            go.Figure: A Plotly Figure
        Notes:
            Overloading this method is optional, some components don't use Plotly figures.
        """
        raise NotImplementedError("This component doesn't use Plotly figures")

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
    def waiting_figure():
        """This helper method creates a waiting figure for the component"""
        waiting_figure = go.Figure()
        waiting_figure.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text="Waiting for data...", showarrow=False, font=dict(size=32)
        )
        waiting_figure.update_layout(
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        return waiting_figure
