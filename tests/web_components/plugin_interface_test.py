"""Tests if plugin interface works with correct and incorrect inputs and returns"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginType, PluginInputType
from sageworks.api.model import Model


class CorrectPlugin(PluginInterface):
    """Subclass of PluginInterface with correct inputs and returns."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, model: Model) -> go.Figure:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
             model (Model): An instantiated Model object
        Returns:
             go.Figure: A Plotly Figure object
        """
        return PluginInterface.message_figure("I'm a good plugin...")


class IncorrectMethods(PluginInterface):
    """Subclass of PluginInterface with incorrect methods
    they have create_component but forgot to implement generate_component_figure"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

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
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    # Component is an incorrectly named keyword argument
    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, model: list) -> go.Figure:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            model (list): An incorrect argument type
        Returns:
            go.Figure: A Figure object containing the confusion matrix.
        """
        return PluginInterface.message_figure("I'm a bad plugin...")


class IncorrectReturnType(PluginInterface):
    """Subclass of PluginInterface with incorrect return type."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, model: Model) -> list:
        """Create a Figure but give the wrong return type.
        Args:
            model (Model): An instantiated Model object
        Returns:
            list: An incorrect return type
        """
        return [1, 2, 3]  # Incorrect return type


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
