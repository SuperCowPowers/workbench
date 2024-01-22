"""A Singleton Plugin Manager Class: Manages the loading and retrieval of SageWorks plugins"""
import os
import logging
import importlib
from typing import Dict, List, Any

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
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
        self.plugins: Dict[str, List[Any]] = {"web_components": [], "transforms": [], "views": [], "pages": []}
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
            self._load_plugin_type(self.plugin_dir, plugin_type)

    def _load_plugin_type(self, base_dir: str, plugin_type: str):
        """
        Load plugins of a specific type from a subdirectory.

        Args:
            base_dir (str): Base directory of the plugins.
            plugin_type (str): Type of the plugin to load (e.g., 'transforms').
        """
        type_dir = os.path.join(base_dir, plugin_type)
        if not os.path.isdir(type_dir):
            return

        for filename in os.listdir(type_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                file_path = os.path.join(type_dir, filename)
                module_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # Check if the attribute is a class and is defined in the module
                        if isinstance(attr, type) and attr.__module__ == module.__name__:
                            # For web components, check if the class is a subclass of WebPluginInterface
                            if plugin_type == "web_components" and issubclass(attr, WebPluginInterface):
                                self.log.important(f"Loading {plugin_type} plugin: {attr_name}")
                                self.plugins[plugin_type].append(attr)
                            else:
                                # We'll add some logic here for the other types
                                pass

    def get_plugins(self, plugin_type: str, web_plugin_type: WebPluginType = None) -> List[Any]:
        """
        Retrieve a list of plugins of a specific type.

        Args:
            plugin_type (str): The type of plugins to retrieve.
            web_plugin_type (WebPluginType): The type of web plugin to retrieve (if applicable).

        Returns:
            List[Any]: A list of INSTANTIATED plugin classes of the requested type.
        """
        if WebPluginType is None:
            plugin_classes = self.plugins.get(plugin_type, [])
            return [x() for x in plugin_classes]
        else:
            plugin_classes = [x for x in self.plugins.get(plugin_type, []) if x.plugin_type == web_plugin_type]
            return [x() for x in plugin_classes]


if __name__ == "__main__":
    """Exercise the PluginManager class"""

    # Create the class, load plugins, and call various methods
    manager = PluginManager()
    transforms = manager.get_plugins("web_components")

    # Get web components for the model view
    print(manager.get_plugins("web_components", WebPluginType.MODEL))

    # Get web components for the endpoint view
    print(manager.get_plugins("web_components", WebPluginType.ENDPOINT))
