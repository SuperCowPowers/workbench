"""A Singleton Plugin Manager Class: Manages the loading and retrieval of Workbench plugins"""

import os
import sys
import atexit
import shutil
import tempfile
import logging
import importlib
import threading
from typing import Union, Dict, List, Any, Optional
from types import ModuleType

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import copy_s3_files_to_local
from workbench.web_interface.page_views.page_view import PageView
from workbench.web_interface.components.plugin_interface import PluginInterface
from workbench.web_interface.components.plugin_interface import PluginPage


class PluginManager:
    """A Singleton Plugin Manager Class: Manages the loading and retrieval of Workbench plugins"""

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

        self.log = logging.getLogger("workbench")
        self.config_plugin_dir = None
        self.loading_dir = False
        self.plugin_modified_time = None
        self.plugins: Dict[str, dict] = {"components": {}, "transforms": {}, "views": {}, "pages": {}, "css": {}}

        # Get the plugin directory from the config
        cm = ConfigManager()
        self.config_plugin_dir = cm.get_config("WORKBENCH_PLUGINS")
        if not self.config_plugin_dir:
            self.log.warning("WORKBENCH_PLUGINS not set. No plugins will be loaded.")
            return

        # Load the plugins
        self.load_plugins()

        # Singleton is now initialized
        self.__initialized = True

    def load_plugins(self):
        """Loads plugins from our 'load_from' directory"""

        # Check if the config directory is local file or S3
        if not self.config_plugin_dir.startswith("s3://"):
            self.loading_dir = self.config_plugin_dir

        # Load plugins from S3 (copy to a temporary directory, then load)
        else:
            self.loading_dir = tempfile.mkdtemp()
            copy_s3_files_to_local(self.config_plugin_dir, self.loading_dir)

            # Cleanup the temporary directory on exit
            atexit.register(self._cleanup_temp_dir)

        # Add the loading directory to the PYTHONPATH for custom packages
        sys.path.append(os.path.join(self.loading_dir, "packages"))

        self.log.important(f"Loading plugins from {self.loading_dir}...")
        for plugin_type in self.plugins.keys():
            self._load_plugins(self.loading_dir, plugin_type)

        # Store the most recent modified time
        self.plugin_modified_time = self._most_recent_modified_time()

    def _load_plugins(self, base_dir: str, plugin_type: str):
        """Internal: Load plugins of a specific type from a subdirectory.

        Args:
            base_dir (str): Base directory of the plugins.
            plugin_type (str): Type of the plugin to load (e.g., components, pages, transforms).
        """
        plugin_dir = os.path.join(base_dir, plugin_type)
        if not os.path.isdir(plugin_dir):
            return

        # For every file in the plugin directory
        for filename in os.listdir(plugin_dir):

            # Normal plugin loading
            module = self._load_module(plugin_dir, filename)
            if module is None:
                self.log.warning(f"Failed to load plugin: '{filename}' skipping...")
                continue

            # Now we have a module, let's iterate through its attributes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if the attribute is a class and is defined in the module
                if isinstance(attr, type) and attr.__module__ == module.__name__:
                    self.log.important(f"Loading {plugin_type} plugin: {attr_name}")

                    # For web components, check if the class is a subclass of PluginInterface
                    if plugin_type == "components":
                        if issubclass(attr, PluginInterface):
                            self.plugins[plugin_type][attr_name] = attr
                        else:
                            # PluginInterface has additional information for failed validation
                            valid, validation_error = PluginInterface.validate_subclass(attr)
                            self.log.error(f"Plugin '{attr_name}' failed validation:")
                            self.log.error(f"\tFile: {os.path.join(plugin_dir, filename)}")
                            self.log.error(f"\tClass: {attr_name}")
                            self.log.error(f"\tDetails: {filename} {validation_error}")

                    # For views, check if the class is a subclass of PageView
                    elif plugin_type == "views" and issubclass(attr, PageView):
                        self.plugins[plugin_type][attr_name] = attr

                    # For pages, check if the class has the required page plugin method (page_setup)
                    elif plugin_type == "pages":
                        if hasattr(attr, "page_setup"):
                            self.plugins[plugin_type][attr_name] = attr
                        else:
                            self.log.warning(f"Class {attr_name} in {filename} does not have all required page methods")

                    # Unexpected type
                    else:
                        self.log.error(f"Unexpected plugin type '{plugin_type}' for plugin '{attr_name}'")

    def _load_module(self, dir_path: str, filename: str) -> Optional[ModuleType]:
        """Internal: Load a module from a file"""
        try:
            if filename.endswith(".py") and not filename.startswith("_"):
                file_path = os.path.join(dir_path, filename)
                module_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
        except Exception as e:
            # Log or handle the exception as needed
            self.log.critical(f"Failed to load plugin: '{filename}': {e}")
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

    def get_list_of_web_plugins(self, plugin_page: PluginPage = None) -> List[Any]:
        """
        Retrieve a list of plugins for a specific page (e.g., MODEL, ENDPOINT).

        Args:
            plugin_page (PluginPage): Get plugins for a specific page (e.g., MODEL, ENDPOINT).

        Returns:
            List[Any]: A list of INSTANTIATED plugin classes for the requested page.
        """
        plugin_classes = [
            self.plugins["components"][x]
            for x in self.plugins["components"]
            if self.plugins["components"][x].auto_load_page == plugin_page
        ]
        return [x() for x in plugin_classes]

    def get_web_plugin(self, plugin_name: str) -> PluginInterface:
        """
        Retrieve a specific web plugin by name

        Args:
            plugin_name (str): The name of the web plugin to retrieve

        Returns:
            PluginInterface: The INSTANTIATED web plugin class with the given name
        """
        web_plugin = self.plugins["components"].get(plugin_name)
        return web_plugin() if web_plugin else None

    def get_view(self, view_name: str) -> Union[PageView, None]:
        """
        Retrieve a page view plugin with the given name.

        Args:
            view_name (str): The name of the page view to retrieve (None if not found).

        Returns:
            PageView: An INSTANTIATED view class with the given name.
        """
        view = self.plugins["views"].get(view_name)
        return view() if view else None

    def get_pages(self) -> dict:
        """
        Retrieve a dict of plugins pages

        Returns:
            dict: A dict of INSTANTIATED plugin pages.
        """
        pages = self.plugins["pages"]
        return {name: page() for name, page in pages.items()}

    def get_pages_with_timeout(self) -> dict:
        """
        Retrieve a dict of plugin pages with a timeout mechanism using threads.

        Returns:
            dict: A dict of instantiated plugin pages or excludes pages that take too long.
        """
        pages = self.plugins["pages"]  # Dictionary of page name to page class
        instantiated_pages = {}
        timeout_seconds = 10  # Timeout threshold

        def instantiate_page(name, page_func):
            """Attempt to instantiate a page and log errors if it fails or times out."""
            result = [None]
            thread = threading.Thread(target=lambda: result.__setitem__(0, page_func()))
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                self.log.critical(f"Page {name} load longer than {timeout_seconds} seconds...")
            else:
                instantiated_pages[name] = result[0]

        # Instantiate each page with the timeout mechanism
        for name, page in pages.items():
            try:
                instantiate_page(name, page)
            except Exception as e:
                logging.critical(f"Error instantiating page '{name}': {e}")

        return instantiated_pages

    def _cleanup_temp_dir(self):
        """Cleans up the temporary directory created for S3 files."""
        if self.loading_dir and os.path.isdir(self.loading_dir):
            self.log.important(f"Cleaning up temporary directory: {self.loading_dir}")
            shutil.rmtree(self.loading_dir)
            self.loading_dir = None

    def plugins_modified(self) -> bool:
        """Check if the plugins have been modified since the last check

        Returns:
            bool: True if the plugins have been modified, else False
        """
        most_recent_time = self._most_recent_modified_time()
        if most_recent_time > self.plugin_modified_time:
            self.log.important("Plugins have been modified")
            self.plugin_modified_time = most_recent_time
            return True
        return False

    # Helper method to compute the most recent modified time
    def _most_recent_modified_time(self) -> float:
        """Internal: Compute the most recent modified time of a directory"""
        most_recent_time = 0
        for root, _, files in os.walk(self.loading_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_modified_time = os.path.getmtime(file_path)
                if file_modified_time > most_recent_time:
                    most_recent_time = file_modified_time
        return most_recent_time

    def __repr__(self) -> str:
        """String representation of the PluginManager state and contents

        Returns:
            str: String representation of the PluginManager state and contents
        """
        summary = "WORKBENCH_PLUGINS: " + self.config_plugin_dir + "\n"
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

    # Get web components for the model page
    model_plugin = manager.get_list_of_web_plugins(PluginPage.MODEL)

    # Get web components for the endpoint page
    endpoint_plugin = manager.get_list_of_web_plugins(PluginPage.ENDPOINT)

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

    # Test Modified Time by modifying a plugin file
    cm = ConfigManager()
    plugin_path = cm.get_config("WORKBENCH_PLUGINS") + "/components/endpoint_turbo.py"
    print(manager.plugins_modified())
    # Modify the plugin file modified time to now
    os.utime(plugin_path, None)
    print(manager.plugins_modified())

    # Test S3 Plugin Loading
    print("\n\n*** Testing S3 Plugin Loading ***\n\n")
    s3_path = "s3://sandbox-workbench-artifacts/workbench_plugins"
    cm = ConfigManager()
    cm.set_config("WORKBENCH_PLUGINS", s3_path)

    # Since we're using a singleton, we need to create a new instance
    PluginManager._instance = None
    manager = PluginManager()

    # Get web components for the model view
    model_plugin = manager.get_list_of_web_plugins(PluginPage.MODEL)

    # Get web components for the endpoint view
    endpoint_plugin = manager.get_list_of_web_plugins(PluginPage.ENDPOINT)

    # Get view plugin
    my_view = manager.get_view("ModelPluginView")

    # Get plugin pages
    plugin_pages = manager.get_pages()

    # Get all the plugins
    pprint(manager.get_all_plugins())
