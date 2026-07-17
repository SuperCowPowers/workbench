from abc import abstractmethod
from typing import Union, Tuple
from enum import Enum
import logging
from dash import no_update
from dash.development.base_component import Component

# Local Imports
from workbench.web_interface.components.component_interface import ComponentInterface
from workbench.utils.theme_manager import ThemeManager

log = logging.getLogger("workbench")


class PluginPage(Enum):
    """Plugin Page: Specify which page will AUTO load the plugin (CUSTOM/NONE = Don't autoload)"""

    DATA_SOURCE = "data_source"
    FEATURE_SET = "feature_set"
    MODEL = "model"
    ENDPOINT = "endpoint"
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

    # Shared ThemeManager instance for all plugins
    theme_manager = ThemeManager()

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

    def set_theme(self, theme: str) -> list:
        """Called when the application theme changes. Override to re-render with new theme colors.

        The default implementation returns no_update for all properties. Plugins that have
        theme-dependent rendering (e.g., figures with colorscales) should override this method
        to re-render their components.

        Args:
            theme (str): The name of the new theme (e.g., "light", "dark", "midnight_blue").

        Returns:
            list: Updated property values (same format as update_properties), or no_update list.

        Example:
            def set_theme(self, theme: str) -> list:
                if self.model is None:
                    return [no_update] * len(self.properties)
                return self.update_properties(self.model)
        """
        return [no_update] * len(self.properties)

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

    # issubclass(x, PluginInterface) is used as the "is this a valid plugin?" gate,
    # so the hook delegates to validate_subclass and returns just the boolean.
    @classmethod
    def __subclasshook__(cls, subclass) -> Union[bool, type(NotImplemented)]:
        if cls is PluginInterface:
            return cls.validate_subclass(subclass)[0]
        return NotImplemented

    @classmethod
    def validate_subclass(cls, subclass) -> Tuple[bool, str]:
        """Validate that a subclass satisfies the plugin contract.

        A plugin is valid when it defines the required attributes and provides a
        concrete implementation of every abstract method -- directly or inherited
        from a parent plugin. Returns (True, reason) or (False, reason).
        """
        # Required attributes (normally enforced by __init_subclass__, re-checked here
        # since validate_subclass may be called on a class that isn't a real subclass)
        for attr in ("auto_load_page", "plugin_input_type"):
            if not hasattr(subclass, attr):
                return False, f"Missing required attribute '{attr}' in {subclass.__name__}"

        # Every abstract method must resolve to a concrete implementation. An
        # unimplemented method keeps __isabstractmethod__ = True anywhere in the MRO,
        # so an inherited implementation (subclassing another plugin) passes.
        for method in cls.__abstractmethods__:
            impl = getattr(subclass, method, None)
            if impl is None or getattr(impl, "__isabstractmethod__", False):
                return False, f"Method '{method}' is not implemented by {subclass.__name__}"

        return True, "Subclass validation successful"
