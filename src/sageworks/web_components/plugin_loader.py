import os
import importlib.util
from typing import List
import logging
import inspect

# SageWorks imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginType
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger("sageworks")


def load_plugins_from_dir(directory: str, plugin_type: PluginType) -> List[PluginInterface]:
    """Load all the plugins from the given directory.
    Args:
        directory (str): The directory to load the plugins from.
    Returns:
        List[PluginInterface]: A list of plugins that were loaded.
    """

    # Sanity check
    if directory is None or not os.path.isdir(directory):
        log.info(f"Not loading plugins, from directory: {directory}")
        return []

    plugins = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)

            # Load the module
            spec = importlib.util.spec_from_file_location(filename, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for classes that inherit from PluginInterface
            class_found = False
            for attribute_name, attribute in inspect.getmembers(module, inspect.isclass):
                if attribute.__module__ == module.__name__:  # This filters out imported classes
                    class_found = True
                    if issubclass(attribute, PluginInterface) and attribute is not PluginInterface:
                        instance = attribute()
                        if instance.plugin_type == plugin_type:  # Check if plugin_type matches
                            plugins.append(instance)
                    else:
                        log.error(f"The class {attribute_name} in {filename} does not inherit from PluginInterface.")
            if not class_found:
                log.error(f"No classes found in {filename}.")

    return plugins


if __name__ == "__main__":
    # This is a simple example of how to load plugins from a directory
    plugin_dir = os.getenv("SAGEWORKS_PLUGINS")
    plugins = load_plugins_from_dir(plugin_dir)
    for plugin in plugins:
        log.info(f"Loaded plugin: {plugin.__class__.__name__}")
