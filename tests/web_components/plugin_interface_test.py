"""Tests the PluginInterface subclass validation (correct, incomplete, and inherited plugins)"""

from dash import dcc

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.api.model import Model


class CorrectPlugin(PluginInterface):
    """A proper plugin that implements both required methods."""

    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        self.container = dcc.Graph(id=component_id, figure=self.waiting_figure())
        return self.container

    def update_properties(self, model: Model) -> list:
        text_figure = PluginInterface.display_text("I'm a good plugin...")
        return [text_figure]


class IncompletePlugin(PluginInterface):
    """Implements create_component but forgets update_properties."""

    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        return dcc.Graph(id=component_id, figure=self.waiting_figure())


class NeverImplemented(PluginInterface):
    """Implements neither required method."""

    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL


class SubclassPlugin(CorrectPlugin):
    """Specializes an existing plugin, inheriting both required methods (overriding neither)."""

    def __init__(self):
        super().__init__()


def test_correct_plugin():
    """A plugin implementing both required methods is a valid subclass."""
    assert issubclass(CorrectPlugin, PluginInterface) is True


def test_incomplete_plugin():
    """A plugin missing one required method is rejected."""
    assert issubclass(IncompletePlugin, PluginInterface) is False


def test_never_implemented():
    """A plugin implementing no required methods is rejected."""
    assert issubclass(NeverImplemented, PluginInterface) is False


def test_subclassing_a_plugin():
    """A plugin that subclasses another plugin (inheriting both methods) is valid.

    Regression: an inherited implementation used to be rejected because validation
    compared method qualnames to the subclass name instead of checking implementation.
    """
    assert issubclass(SubclassPlugin, PluginInterface) is True
    valid, reason = PluginInterface.validate_subclass(SubclassPlugin)
    assert valid is True, reason


if __name__ == "__main__":
    test_correct_plugin()
    test_incomplete_plugin()
    test_never_implemented()
    test_subclassing_a_plugin()
