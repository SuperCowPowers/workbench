from abc import abstractmethod
import inspect
from typing import Union, Tuple, get_args, get_origin
from enum import Enum
import logging
from dash.development.base_component import Component

# Local Imports
from workbench.web_interface.components.component_interface import ComponentInterface

log = logging.getLogger("workbench")


class PluginPage(Enum):
    """Plugin Page: Specify which page will AUTO load the plugin (CUSTOM/NONE = Don't autoload)"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
    PIPELINE = "pipeline"
    GRAPH = "graph"
    COMPOUND = "compound"
    CUSTOM = "custom"
    NONE = "none"


class PluginInputType(Enum):
    """Plugin Input Type: Specify the type of object that the plugin will receive as input"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
    PIPELINE = "pipeline"
    GRAPH = "graph"
    DATAFRAME = "dataframe"
    COMPOUND = "compound"
    CUSTOM = "custom"


class PluginInterface(ComponentInterface):
    """A Web Plugin Interface

    Notes:
        - The 'create_component' method must be implemented by the child class
        - The 'update_properties' method must be implemented by the child class
        - The 'register_internal_callbacks' method is optional
    """

    @abstractmethod
    def create_component(self, component_id: str) -> Component:
        """Create a Dash Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            Component: A Dash Base Component.
        """
        pass

    @abstractmethod
    def update_properties(self, data_object: ComponentInterface.WorkbenchObject, **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            data_object (ComponentInterface.WorkbenchObject): The instantiated data object for the plugin type.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).

        Returns:
            list: A list of the updated property values for the plugin.
        """
        pass

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin.

        Notes:
            Implementing this method is optional. This method is useful for registering internal callbacks
            that are specific to the plugin. For example, a plugin that needs to update some properties based
            on a dropdown selection.
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
    # and signatures. It returns True, False, or NotImplemented to support issubclass checks.
    @classmethod
    def __subclasshook__(cls, subclass) -> Union[bool, type(NotImplemented)]:
        if cls is PluginInterface:
            # Check if the subclass has all the required attributes
            if not all(hasattr(subclass, attr) for attr in ("auto_load_page", "plugin_input_type")):
                log.error(f"Missing required attributes in {subclass.__name__}")
                return False

            # Check if the subclass has all the required methods with correct signatures
            required_methods = set(cls.__abstractmethods__)
            for method in required_methods:
                # Check for the presence of the method
                if not hasattr(subclass, method):
                    log.error(f"Missing required method: {method} in {subclass.__name__}")
                    return False

                # Check if the method is implemented by the subclass itself
                subclass_method = getattr(subclass, method)
                if subclass_method.__qualname__.split(".")[0] != subclass.__name__:
                    log.error(f"Method {method} is not implemented by {subclass.__name__}")
                    return False

                # Check argument types and return type
                base_class_method = getattr(cls, method)
                arg_type_error = cls._check_argument_types(base_class_method, subclass_method)
                return_type_error = cls._check_return_type(base_class_method, subclass_method)
                if arg_type_error or return_type_error:
                    log.error(arg_type_error)
                    log.error(return_type_error)
                    return False

            # If all checks pass, return True
            return True
        return NotImplemented

    # Return detailed validation information
    @classmethod
    def validate_subclass(cls, subclass) -> Tuple[bool, str]:
        """Validates the subclass and returns a detailed description of any issues."""

        # Check required attributes
        if not all(hasattr(subclass, attr) for attr in ("auto_load_page", "plugin_input_type")):
            return False, f"Missing required attributes in {subclass.__name__}"

        # Get required abstract methods
        required_methods = set(cls.__abstractmethods__)

        for method in required_methods:
            if not hasattr(subclass, method):
                return False, f"Missing required method: {method} in {subclass.__name__}"

            subclass_method = getattr(subclass, method)

            # Check if method is implemented in the subclass itself
            if subclass_method.__qualname__.split(".")[0] != subclass.__name__:
                # Get the line number for the method
                method_info = inspect.getsourcelines(subclass_method)
                line_number = method_info[1]
                return False, f"Method {method} is not implemented by {subclass.__name__} (line {line_number})"

            # Check argument types
            base_class_method = getattr(cls, method)
            arg_type_error = cls._check_argument_types(base_class_method, subclass_method)
            if arg_type_error:
                method_info = inspect.getsourcelines(subclass_method)
                line_number = method_info[1]
                return False, f"(line {line_number}): {arg_type_error}"

            # Check return type
            return_type_error = cls._check_return_type(base_class_method, subclass_method)
            if return_type_error:
                method_info = inspect.getsourcelines(subclass_method)
                line_number = method_info[1]
                return False, f"(line {line_number}): {return_type_error}"

        return True, "Subclass validation successful"

    @staticmethod
    def _is_subtype_of_any(actual: type, expected_types: tuple) -> bool:
        """Check if a type is a subtype of any expected types."""
        if get_origin(actual) is Union:
            actual_types = get_args(actual)
            return all(any(issubclass(act, exp) for exp in expected_types) for act in actual_types)
        return any(issubclass(actual, exp) for exp in expected_types)

    @classmethod
    def _check_argument_types(cls, base_class_method, subclass_method) -> Union[None, str]:
        """Check that argument types match between base class and subclass methods."""
        expected_arg_types = [v for k, v in base_class_method.__annotations__.items() if k != "return" and k != "self"]
        actual_arg_types = [
            param.annotation for param in inspect.signature(subclass_method).parameters.values() if param.name != "self"
        ]

        for expected, actual in zip(expected_arg_types, actual_arg_types):
            if get_origin(expected) is Union:
                expected_types = get_args(expected)
                if not cls._is_subtype_of_any(actual, expected_types):
                    return f"Argument type {actual} not in ({expected_types})"
            else:
                if expected != actual:
                    return f"Argument {actual} doesn't match expected {expected}"

        return None

    @classmethod
    def _check_return_type(cls, base_class_method, subclass_method) -> Union[None, str]:
        """Check that return types match between base class and subclass methods."""
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
            if not (actual_return_type == list or get_origin(actual_return_type) is list):
                return f"Incorrect return type for {method_name} (expected list, got {actual_return_type.__name__})"

        return None
