"""An abstract class that defines the web component interface for SageWorks"""
from abc import abstractmethod
from typing import Any
from enum import Enum

# Local Imports
from sageworks.web_components.component_interface import ComponentInterface


class PluginType(Enum):
    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"


class PluginInputType(Enum):
    DATA_SOURCE_DETAILS = "data_source_details"
    FEATURE_SET_DETAILS = "feature_set_details"
    MODEL_DETAILS = "model_details"
    ENDPOINT_DETAILS = "endpoint_details"


class PluginInterface(ComponentInterface):
    """A Web Plugin Interface
    Notes:
      - These methods are ^stateless^, all data should be passed through the
        arguments and the implementations should not reference 'self' variables
      - The 'create_component' method must be implemented by the child class
      - The 'generate_component_figure' is optional (some components don't use Plotly figures)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "plugin_type") or not isinstance(cls.plugin_type, PluginType):
            raise TypeError("Subclasses must define a 'plugin_type' of type PluginType")

        if not hasattr(cls, "plugin_input_type") or not isinstance(cls.plugin_input_type, PluginInputType):
            raise TypeError("Subclasses must define a 'plugin_input_type' of type PluginInputType")

    @abstractmethod
    def create_component(self, component_id: str, **kwargs: Any) -> ComponentInterface.ComponentTypes:
        """Create a Dash Component without any data.
        Args:
            component_id (str): The ID of the web component
            kwargs (Any): Any additional arguments to pass to the component
        Returns:
            Union[go.Figure, dcc.Markdown, html.Div]: The Dash Web component
        """
        pass
