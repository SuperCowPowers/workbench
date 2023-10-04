"""An abstract class that defines the web component interface for SageWorks"""

from abc import ABC, abstractmethod
from typing import Any, Union
import plotly.graph_objects as go
import pandas as pd
from dash import dcc, html
from dash import dash_table


class ComponentInterface(ABC):
    """A Stateless Abstract Web Component Interface
    Notes:
      - These methods are ^stateless^,  all data should be passed through the
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
        """
        raise NotImplementedError("This component doesn't use Plotly figures")
