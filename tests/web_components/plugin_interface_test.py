"""Tests if plugin interface works with correct and incorrect inputs and returns"""

from dash import dcc
import plotly.graph_objects as go


# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.api.model import Model


class CorrectPlugin(PluginInterface):
    """Subclass of PluginInterface with correct inputs and returns."""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        self.container = None

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        self.container = dcc.Graph(id=component_id, figure=self.waiting_figure())
        return self.container

    def update_properties(self, model: Model) -> list:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
             model (Model): An instantiated Model object
        """
        text_figure = PluginInterface.display_text("I'm a good plugin...")
        return [text_figure]


class IncorrectMethods(PluginInterface):
    """Subclass of PluginInterface with incorrect methods
    they have create_component but forgot to implement update_properties"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())


class IncorrectArgTypes(PluginInterface):
    """Subclass of PluginInterface with an incorrectly typed argument."""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    # Component is an incorrectly named keyword argument
    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: A Dash Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def update_properties(self, model: list) -> go.Figure:
        """Create a Plotly Figure
        Args:
            model (list): An incorrect argument type
        Returns:
            go.Figure: A Figure object
        """
        return PluginInterface.display_text("I'm a bad plugin...")


class IncorrectReturnType(PluginInterface):
    """Subclass of PluginInterface with incorrect return type."""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def update_properties(self, model: Model) -> go.Figure:
        """Create a Figure but give the wrong return type.
        Args:
            model (Model): An instantiated Model object
        Returns:
            list: An incorrect return type
        """
        return go.Figure()  # Incorrect return type


def test_incorrect_methods():
    """Test if incorrect methods are caught by the PluginInterface"""
    subclass_cond = issubclass(IncorrectMethods, PluginInterface)
    print(f"Incorrect Methods is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is False


def test_incorrect_args():
    """Test if incorrect args are caught by the PluginInterface"""
    subclass_cond = issubclass(IncorrectArgTypes, PluginInterface)
    print(f"Incorrect names is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is False


def test_incorrect_return_type():
    """Test if incorrect types are caught by the PluginInterface"""
    subclass_cond = issubclass(IncorrectReturnType, PluginInterface)
    print(f"Incorrect return type is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is False


def test_correct_plugin():
    """Test if a proper subclass of PluginInterface returns True"""
    subclass_cond = issubclass(CorrectPlugin, PluginInterface)
    print(f"Correct is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is True


if __name__ == "__main__":
    # Run the tests
    test_incorrect_methods()
    test_incorrect_args()
    test_incorrect_return_type()
    test_correct_plugin()
