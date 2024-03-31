"""An abstract class that defines the web component interface for SageWorks"""

from abc import abstractmethod
from inspect import signature
from typing import Union, get_args
from enum import Enum

# Local Imports
from sageworks.web_components.component_interface import ComponentInterface


class PluginPage(Enum):
    """Plugin Page: Specify which page will autoload the plugin (CUSTOM = Don't autoload)"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
    CUSTOM = "custom"


class PluginInputType(Enum):
    """Plugin Input Type: Specify the type of object that the plugin will receive as input"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"


class PluginInterface(ComponentInterface):
    """A Web Plugin Interface
    Notes:
      - These methods are ^stateless^, all data should be passed through the
        arguments and the implementations should not reference 'self' variables
      - The 'create_component' method must be implemented by the child class
      - The 'update_contents' method must be implemented by the child class
    """

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
    def update_contents(self, data_object: ComponentInterface.SageworksObject) -> ComponentInterface.ContentTypes:
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

        # Ensure the subclass defines the required plugin_page and plugin_input_type
        if not hasattr(cls, "plugin_page") or not isinstance(cls.plugin_page, PluginPage):
            raise TypeError("Subclasses must define a 'plugin_page' of type PluginPage")
        if not hasattr(cls, "plugin_input_type") or not isinstance(cls.plugin_input_type, PluginInputType):
            raise TypeError("Subclasses must define a 'plugin_input_type' of type PluginInputType")

    # If any base class method or parameter is missing from a subclass, or if a subclass method parameter is not
    # correctly typed a call of issubclass(subclass, cls) will return False, allowing runtime checks for plugins
    # The plugin loader calls issubclass(subclass, cls) to determine if the subclass is a valid plugin
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is PluginInterface:
            # Check if the subclass has all the required attributes
            if not all(hasattr(subclass, attr) for attr in ("plugin_page", "plugin_input_type")):
                cls.log.warning(f"Subclass {subclass.__name__} is missing required attributes")
                return False

            # Check if the subclass has all the required methods with correct signatures
            required_methods = set(cls.__abstractmethods__)
            for method in required_methods:
                # Check for the presence of the method
                if not hasattr(subclass, method):
                    cls.log.warning(f"Subclass {subclass.__name__} is missing required method {method}")
                    return False

                # Check if the method is implemented by the subclass itself
                subclass_method = getattr(subclass, method)
                if subclass_method.__qualname__.split(".")[0] != subclass.__name__:
                    cls.log.warning(f"Subclass {subclass.__name__} has not implemented the method {method}")
                    return False

                # Check argument types
                base_class_method = getattr(cls, method)
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
