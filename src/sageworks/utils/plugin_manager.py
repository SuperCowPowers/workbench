"""A Singleton Plugin Manager Class: Manages the loading and retrieval of SageWorks plugins"""

import os
import logging
import importlib
from typing import Union, Dict, List, Any

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
from sageworks.views.view import View
from sageworks.web_components.plugin_interface import PluginInterface as WebPluginInterface
from sageworks.web_components.plugin_interface import PluginType as WebPluginType


class PluginManager:
    """A Singleton Plugin Manager Class: Manages the loading and retrieval of SageWorks plugins"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the class if it doesn't exist, else return the existing instance."""
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance.__initialized = False  # Gets initialized in __init__
        return cls._instance

    def __init__(self):
        """Initialize the PluginManager as a singleton."""
        if self.__initialized:
            return

        self.log = logging.getLogger("sageworks")
        self.plugins: Dict[str, dict] = {"web_components": {}, "transforms": {}, "views": {}, "pages": {}}
        cm = ConfigManager()
        self.plugin_dir = cm.get_config("SAGEWORKS_PLUGINS")
        if not self.plugin_dir:
            self.log.warning("SAGEWORKS_PLUGINS not set. No plugins will be loaded.")
            return
        self.load_plugins()
        self.__initialized = True

    def load_plugins(self):
        """Loads plugins from our plugins directory"""
        self.log.important(f"Loading plugins from {self.plugin_dir}...")
        for plugin_type in self.plugins.keys():
            self._load_plugins(self.plugin_dir, plugin_type)

    def _load_plugins(self, base_dir: str, plugin_type: str):
        """Internal: Load plugins of a specific type from a subdirectory.

        Args:
            base_dir (str): Base directory of the plugins.
            plugin_type (str): Type of the plugin to load (e.g., web_components, pages, transforms).
        """
        type_dir = os.path.join(base_dir, plugin_type)
        if not os.path.isdir(type_dir):
            return

        for filename in os.listdir(type_dir):
            module = self._load_module(type_dir, filename)
            if module:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if the attribute is a class and is defined in the module
                    if isinstance(attr, type) and attr.__module__ == module.__name__:
                        # For web components, check if the class is a subclass of WebPluginInterface
                        if plugin_type == "web_components" and issubclass(attr, WebPluginInterface):
                            self.log.important(f"Loading {plugin_type} plugin: {attr_name}")
                            self.plugins[plugin_type][attr_name] = attr

                        # For views, check if the class is a subclass of View
                        elif plugin_type == "views" and issubclass(attr, View):
                            self.log.important(f"Loading {plugin_type} plugin: {attr_name}")
                            self.plugins[plugin_type][attr_name] = attr

                        # For pages, check if the class has the required page plugin method (page_setup)
                        elif plugin_type == "pages":
                            if hasattr(attr, "page_setup"):
                                self.log.important(f"Loading page plugin: {attr_name}")
                                self.plugins[plugin_type][attr_name] = attr
                            else:
                                self.log.warning(
                                    f"Class {attr_name} in {filename} does not have all required page methods"
                                )

                        # Unexpected type
                        else:
                            self.log.warning(f"Class {attr_name} in {filename} invalid {plugin_type} plugin")

    @staticmethod
    def _load_module(dir_path: str, filename: str):
        """Internal: Load a module from a file"""
        if filename.endswith(".py") and not filename.startswith("_"):
            file_path = os.path.join(dir_path, filename)
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        return None

    def get_all_plugins(self) -> Dict[str, dict[Any]]:
        """
        Retrieve a dictionary of all plugins.

        Returns:
            Dict[str, dict]: A dictionary of all plugins.
                {'pages': {name: plugin, name2: plugin2,...
                 'views': {name: plugin, name2: plugin2,...},
        """
        return self.plugins

    def get_list_of_web_plugins(self, web_plugin_type: WebPluginType = None) -> List[Any]:
        """
        Retrieve a list of plugins of a specific type.

        Args:
            web_plugin_type (WebPluginType): The type of web plugin to retrieve (if applicable).

        Returns:
            List[Any]: A list of INSTANTIATED plugin classes of the requested type.
        """
        plugin_classes = [
            self.plugins["web_components"][x]
            for x in self.plugins["web_components"]
            if self.plugins["web_components"][x].plugin_type == web_plugin_type
        ]
        return [x() for x in plugin_classes]

    def get_web_plugin(self, plugin_name: str) -> WebPluginInterface:
        """
        Retrieve a specific web plugin by name

        Args:
            plugin_name (str): The name of the web plugin to retrieve

        Returns:
            WebPluginInterface: The INSTANTIATED web plugin class with the given name
        """
        web_plugin = self.plugins["web_components"].get(plugin_name)
        return web_plugin() if web_plugin else None

    def get_view(self, view_name: str) -> Union[View, None]:
        """
        Retrieve a view plugin with the given name.

        Args:
            view_name (str): The name of the view to retrieve (None if not found).

        Returns:
            View: An INSTANTIATED view class with the given name.
        """
        view = self.plugins["views"].get(view_name)
        return view() if view else None

    def get_pages(self) -> dict[Any]:
        """
        Retrieve a dict of plugins pages

        Returns:
           dict: A dict of INSTANTIATED plugin pages.
        """
        pages = self.plugins["pages"]
        return {name: page() for name, page in pages.items()}

    def __repr__(self) -> str:
        """String representation of the PluginManager state and contents

        Returns:
            str: String representation of the PluginManager state and contents
        """
        summary = "SAGEWORKS_PLUGINS: " + self.plugin_dir + "\n"
        summary += "Plugins:\n"
        for plugin_type, plugin_dict in self.plugins.items():
            for name, plugin in plugin_dict.items():
                summary += f"  {plugin_type}: {name}: {plugin}\n"
        return summary


if __name__ == "__main__":
    """Exercise the PluginManager class"""
    from pprint import pprint

    # Create the class, load plugins, and call various methods
    manager = PluginManager()

    # Get web components for the model view
    model_plugin = manager.get_list_of_web_plugins(WebPluginType.MODEL)

    # Get web components for the endpoint view
    endpoint_plugin = manager.get_list_of_web_plugins(WebPluginType.ENDPOINT)

    # Get view plugin
    my_view = manager.get_view("ModelPluginView")

    # Get plugin pages
    plugin_pages = manager.get_pages()

    # Get all the plugins
    pprint(manager.get_all_plugins())

    # Get a web plugin by name
    my_plugin = manager.get_web_plugin("CustomTurbo")
    print(my_plugin)

    # Test REPR
    print(manager)
