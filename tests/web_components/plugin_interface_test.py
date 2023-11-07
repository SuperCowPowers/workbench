"""Tests if plugin interface works with correct and incorrect inputs and returns"""
from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginType, PluginInputType


class CorrectPlugin(PluginInterface):
    """Subclass of PluginInterface with correct inputs and returns."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> PluginInterface.ComponentTypes:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, figure_input: PluginInputType) -> PluginInterface.FigureTypes:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            figure_input (PluginInputType.MODEL_DETAILS): Model class details attribute
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """

        return PluginInterface.message_figure("I'm a good plugin...")


class IncorrectMethods(PluginInterface):
    """Subclass of PluginInterface with incorrect methods
    they have create_component but forgot to implement generate_component_figure"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> PluginInterface.ComponentTypes:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())


class IncorrectNamedInputs(PluginInterface):
    """Subclass of PluginInterface with incorrectly named inputs."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    # Component is an incorrectly named keyword argument
    def create_component(self, bleh_component: str) -> PluginInterface.ComponentTypes:
        """Create a Confusion Matrix Component without any data.
        Args:
            bleh_component (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=bleh_component, figure=self.waiting_figure())

    def generate_component_figure(self, bleh_figure_input: PluginInputType) -> PluginInterface.FigureTypes:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            bleh_figure_input (PluginInputType): Model class details attribute
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """
        return PluginInterface.message_figure("I'm a bad plugin...")


class IncorrectTypedInputs(PluginInterface):
    """Subclass of PluginInterface with incorrectly typed inputs."""

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

    # figure_input: dict is incorrectly typed (PluginInputType)
    def generate_component_figure(self, figure_input: dict) -> go.Figure:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            figure_input (dict): Model class details attribute
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """
        return PluginInterface.message_figure("I'm a bad plugin...")


def test_incorrect_methods():
    """Test if incorrect methods are caught by the PluginInterface"""
    subclass_cond = issubclass(IncorrectMethods, PluginInterface)
    print(f"Incorrect Methods is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is False


def test_incorrect_names():
    """Test if incorrect names are caught by the PluginInterface"""
    subclass_cond = issubclass(IncorrectNamedInputs, PluginInterface)
    print(f"Incorrect names is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is False


def test_incorrect_types():
    """Test if incorrect types are caught by the PluginInterface"""
    subclass_cond = issubclass(IncorrectTypedInputs, PluginInterface)
    print(f"Incorrect types is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is False


def test_proper_subclass():
    """Test if a proper subclass of PluginInterface returns True"""
    subclass_cond = issubclass(CorrectPlugin, PluginInterface)
    print(f"Correct is a subclass of PluginInterface?: {subclass_cond}")
    print("\n")
    assert subclass_cond is True


if __name__ == "__main__":
    # Run the tests
    test_incorrect_methods()
    test_incorrect_names()
    test_incorrect_types()
    test_proper_subclass()
