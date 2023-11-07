"""An abstract class that defines the web component interface for SageWorks"""
from abc import abstractmethod
import inspect
from enum import Enum

# Local Imports
from sageworks.web_components.component_interface import ComponentInterface


class PluginType(Enum):
    """TBD: Nice Docstring here or link to docs"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"


class PluginInputType(Enum):
    """TBD: Nice Docstring here or link to docs"""

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
      - The 'generate_component_figure' method must be implemented by the child class
    """

    @abstractmethod
    def create_component(self, component_id: str) -> ComponentInterface.ComponentTypes:
        """Create a Dash Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            Union[dcc.Graph, dash_table.DataTable, html.Div] The Dash Web component
        """
        pass

    @abstractmethod
    def generate_component_figure(self, figure_input: PluginInputType) -> ComponentInterface.FigureTypes:
        """Generate a figure from the data in the given dataframe.
        Args:
            figure_input (PluginInputType): The type of input data for the figure.
        Returns:
            Union[dcc.Graph, dash_table.DataTable, html.Div] The Dash Web component
        """
        pass

    #
    # Internal Methods: These methods are used to validate the plugin interface at runtime
    #
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "plugin_type") or not isinstance(cls.plugin_type, PluginType):
            raise TypeError("Subclasses must define a 'plugin_type' of type PluginType")

        if not hasattr(cls, "plugin_input_type") or not isinstance(cls.plugin_input_type, PluginInputType):
            raise TypeError("Subclasses must define a 'plugin_input_type' of type PluginInputType")

    # If any base class method or parameter is missing from a subclass, or if a subclass method parameter is not
    # correctly typed a call of issubclass(subclass, cls) will return False, allowing runtime checks for plugins
    # The plugin loader calls issubclass(subclass, cls) to determine if the subclass is a valid plugin
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is PluginInterface:
            # Check if the subclass has all the required attributes
            if not all(hasattr(subclass, attr) for attr in ("plugin_type", "plugin_input_type")):
                cls.log.warning(f"Subclass {subclass.__name__} is missing required attributes")
                return False

            # Check if the subclass has all the required methods with correct signatures
            required_methods = set(cls.__abstractmethods__)
            for method in required_methods:
                # Check for the presence of the method
                if not hasattr(subclass, method):
                    cls.log.warning(f"Subclass {subclass.__name__} is missing required method {method}")
                    return False

                # Check if the method is different from the base class (i.e., it's been implemented)
                if getattr(subclass, method) is getattr(PluginInterface, method):
                    cls.log.warning(f"Subclass {subclass.__name__} has not implemented the method {method}")
                    return False

                # Check for the correct signature
                expected_signature = inspect.signature(getattr(cls, method))
                actual_signature = inspect.signature(getattr(subclass, method))
                if expected_signature != actual_signature:
                    cls.log.warning(f"Subclass {subclass.__name__} has incorrect signature for method {method}")
                    return False

            # If all checks pass, return True
            return True
        return NotImplemented
