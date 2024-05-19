"""An abstract class that defines the web component interface for SageWorks"""

from abc import abstractmethod
from inspect import signature
from typing import Union, get_args
from enum import Enum
from dash.development.base_component import Component

# Local Imports
from sageworks.web_components.component_interface import ComponentInterface


class PluginPage(Enum):
    """Plugin Page: Specify which page will AUTO load the plugin (CUSTOM/NONE = Don't autoload)"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
    PIPELINE = "pipeline"
    CUSTOM = "custom"
    NONE = "none"


class PluginInputType(Enum):
    """Plugin Input Type: Specify the type of object that the plugin will receive as input"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
    PIPELINE = "pipeline"
    MODEL_TABLE = "model_table"
    PIPELINE_TABLE = "pipeline_table"
    CUSTOM = "custom"


class PluginInterface(ComponentInterface):
    """A Web Plugin Interface
    Notes:
      - The 'create_component' method must be implemented by the child class
      - The 'update_properties' method must be implemented by the child class
    """

    @abstractmethod
    def create_component(self, component_id: str) -> Component:
        """Create a Dash Component without any data.

        Args:
            component_id (str): The ID of the web component

        Returns:
           Component: A Dash Base Component
        """
        pass

    @abstractmethod
    def update_properties(self, data_object: ComponentInterface.SageworksObject, **kwargs) -> list:
        """Update the property values for the plugin component
        Args:
            data_object (sageworks_object): The instantiated data object for the plugin type.
            **kwargs: Additional keyword arguments (plugins can define their own arguments)
        Returns:
            list: A list of the updated properties values for the plugin
        """
        pass

    #
    # Internal Methods: These methods are used to validate the plugin interface at runtime
    #
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ensure the subclass defines the required auto_load_page and plugin_input_type
        if not hasattr(cls, "auto_load_page") or not isinstance(cls.auto_load_page, PluginPage):
            raise TypeError("Subclasses must define a 'auto_load_page' of type PluginPage")
        if not hasattr(cls, "plugin_input_type") or not isinstance(cls.plugin_input_type, PluginInputType):
            raise TypeError("Subclasses must define a 'plugin_input_type' of type PluginInputType")

    # This subclass check ensures that a subclass of PluginInterface has all required attributes, methods,
    # and signatures. It returns False any thing is incorrect enabling runtime validation for plugins.
    # The plugin loader uses issubclass(subclass, cls) to verify plugin subclasses.
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is PluginInterface:
            # Check if the subclass has all the required attributes
            if not all(hasattr(subclass, attr) for attr in ("auto_load_page", "plugin_input_type")):
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
        method_name = base_class_method.__name__
        actual_return_type = subclass_method.__annotations__.get("return", None)

        if actual_return_type is None:
            return "Missing return type annotation in subclass method."

        if method_name == "create_component":
            if not issubclass(actual_return_type, Component):
                return (
                    f"Incorrect return type for {method_name} (expected Component, got {actual_return_type.__name__})"
                )
        elif method_name == "update_properties":
            if not (actual_return_type == list or (getattr(actual_return_type, "__origin__", None) is list)):
                return f"Incorrect return type for {method_name} (expected list, got {actual_return_type.__name__})"

        return None
