import json
import os
import logging
import importlib.resources as resources
from typing import Any, Dict

# Set up the logger
import sageworks  # noqa: F401 (we need to import this to set up the logger)

log = logging.getLogger("sageworks")


class ConfigManager:
    def __init__(self):
        """Initialize the ConfigManager."""
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration based on the SAGEWORKS_CONFIG environment variable.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """

        # Load configuration from environment variable
        sageworks_config = os.environ.get("SAGEWORKS_CONFIG")
        if sageworks_config is None:
            log.info("SAGEWORKS_CONFIG environment variable not set. Using default configuration.")
            return self.load_default_config()

        # Load configuration from AWS Parameter Store
        if sageworks_config == "parameter_store":
            try:
                log.info("Loading site configuration from AWS Parameter Store...")
                return self.get_config_from_aws_parameter_store()
            except Exception:
                log.error("Could not load configuration from AWS Parameter Store. Using default configuration.")
                return self.load_default_config()

        # Load user-specified configuration file
        try:
            log.info(f"Loading site configuration from {sageworks_config}...")
            with open(sageworks_config, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            log.error(f"Could not load configuration from {sageworks_config}. Using default configuration.")
            return self.load_default_config()

    @staticmethod
    def load_default_config() -> Dict[str, Any]:
        """Load the default configuration from the package resources.

        Returns:
            Dict[str, Any]: Default configuration dictionary.
        """
        with resources.open_text("sageworks.resources", "default_config.json") as file:
            return json.load(file)

    def get_config_from_aws_parameter_store(self) -> Dict[str, Any]:
        """Stub method to fetch configuration from AWS Parameter Store.

        Returns:
            Dict[str, Any]: Configuration dictionary from AWS Parameter Store.
        """
        # TODO: Implement AWS Parameter Store fetching logic
        return {}

    def get_config(self, key: str) -> Any:
        """Get a configuration value by key.

        Args:
            key (str): The configuration key to retrieve.

        Returns:
            Any: The value of the configuration key.
        """
        return self.config.get(key, None)


if __name__ == "__main__":
    """Exercise the ConfigManager class"""
    config_manager = ConfigManager()
    api_key = config_manager.get_config("SAGEWORKS_BUCKET")
    print(f"API Key: {api_key}")
