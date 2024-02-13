"""An abstract class that defines the web component interface for SageWorks"""

from abc import abstractmethod
from inspect import signature
from typing import Union, get_args
from enum import Enum
from functools import wraps

# Local Imports
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.api import DataSource, FeatureSet, Model, Endpoint


class PluginType(Enum):
    """Plugin Types: These are the types of plugins that can be loaded"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
    CUSTOM = "custom"


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

    sageworks_object = Union[DataSource, FeatureSet, Model, Endpoint]

    @abstractmethod
    def create_component(self, component_id: str) -> ComponentInterface.ComponentTypes:
        """Create a Dash Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            Union[dcc.Graph, dash_table.DataTable, dcc.Markdown, html.Div] The Dash Web component
        """
        pass

    @abstractmethod
    def generate_component_figure(self, data_object: sageworks_object) -> ComponentInterface.FigureTypes:
        """Generate a figure from the data in the given dataframe.
        Args:
            data_object (sageworks_object): The instantiated data object for the plugin type.
        Returns:
            Union[go.Figure, str]: A Plotly Figure or a Markdown string
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
                base_class_method = getattr(cls, method)
                subclass_method = getattr(subclass, method)
                if subclass_method is getattr(PluginInterface, method):
                    cls.log.warning(f"Subclass {subclass.__name__} has not implemented the method {method}")
                    return False

                # Check argument types
                arg_type_error = cls._check_argument_types(base_class_method, subclass_method)
                if arg_type_error:
                    cls.log.warning(f"Subclass {subclass.__name__} error in method '{method}': {arg_type_error}")
                    return False

                # Check return type
                return_type_error = cls._check_return_type(base_class_method, subclass_method)
                if return_type_error:
                    cls.log.warning(f"Subclass {subclass.__name__} error in method '{method}': {return_type_error}")
                    return False

            # If all checks pass, return True
            return True
        return NotImplemented

    @classmethod
    def _check_argument_types(cls, base_class_method, subclass_method):
        # Extract expected argument types, excluding 'self'
        expected_arg_types = [v for k, v in base_class_method.__annotations__.items() if k != "return" and k != "self"]
        actual_arg_types = [
            param.annotation for param in signature(subclass_method).parameters.values() if param.name != "self"
        ]

        # Iterate over expected and actual argument types together
        for expected, actual in zip(expected_arg_types, actual_arg_types):
            # If the expected type is a Union, use get_args to extract its arguments
            if getattr(expected, "__origin__", None) is Union:
                expected_types = get_args(expected)
                # Check if the actual type is a subtype of any of the expected types
                if not any(issubclass(actual, exp) for exp in expected_types):
                    return f"Expected argument types {expected_types} do not include the actual argument type {actual}"
            else:
                # Direct comparison for non-Union types
                if expected != actual:
                    return f"Expected argument type {expected} does not match actual argument type {actual}"
        return None

    @classmethod
    def _check_return_type(cls, base_class_method, subclass_method):
        return_annotation = base_class_method.__annotations__["return"]
        expected_return_types = get_args(return_annotation)
        return_type = signature(subclass_method).return_annotation

        if return_type not in expected_return_types:
            return f"Incorrect return type (expected one of {expected_return_types}, got {return_type})"
        return None


# This is a helper decorator to catch errors in plugin methods and return a message figure
def plugin_error_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = f"Plugin Crashed: {e.__class__.__name__}: {e}"
            return ComponentInterface.message_figure(error_info)

    return wrapper
