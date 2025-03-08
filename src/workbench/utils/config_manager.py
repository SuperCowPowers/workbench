# Configuration Manager Class (Singleton Pattern)
import json
import os
import sys
import platform
import logging
import importlib.resources as resources  # noqa: F401 Python 3.9 compatibility
from typing import Any, Dict

# Workbench imports
from workbench.utils.license_manager import LicenseManager
from workbench.utils.execution_environment import running_as_service

# Python 3.9 compatibility
from workbench.utils.resource_utils import get_resource_path


class FatalConfigError(Exception):
    """Exception raised for errors in the configuration."""

    def __init__(self):
        sys.exit(1)


class ConfigManager:
    """A Singleton Configuration Manager Class"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the class if it doesn't exist, else return the existing instance."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.__initialized = False  # Gets initialized in __init__
        return cls._instance

    def __init__(self):
        """Initialize the ConfigManager as a singleton."""
        if self.__initialized:
            return

        self.log = logging.getLogger("workbench")
        self.site_config_path = None
        self.using_default_config = False

        # Load the configuration
        self.config = self._load_config()

        # Load the LicenseManager
        self.license_manager = LicenseManager()

        # Check if we're running as a service
        if running_as_service():
            self.log.important("Running as part of a Service...")

            # Remove the AWS_PROFILE from the config
            if "AWS_PROFILE" in self.config:
                self.log.important("Removing AWS_PROFILE from config...")
                del self.config["AWS_PROFILE"]

            # Overwrite the config with the ENV vars
            self.overwrite_config_with_env()

            # Check AWS Parameter Store for config
            self.overwrite_config_with_parameter_store()

        # AOK
        self.__initialized = True

    def get_config(self, key: str, default_value: Any = None) -> Any:
        """Get a configuration value by key.

        Args:
            key (str): The configuration key to retrieve.
            default_value (Any, optional): The default value to return if not found. Defaults to None.

        Returns:
            Any: The value of the configuration key.
        """
        # Special logic for WORKBENCH_PLUGINS
        if key == "WORKBENCH_PLUGINS":
            plugin_dir = self.config.get(key, default_value)
            if plugin_dir in ["package", "", None]:
                return os.path.join(os.path.dirname(__file__), "../plugins")
            else:
                return plugin_dir

        # Normal logic
        return self.config.get(key, default_value)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dict[str, Any]: All configuration values.
        """
        # Grab all config except the WORKBENCH_API_KEY
        output = {key: value for key, value in self.config.items() if key != "WORKBENCH_API_KEY"}

        # Add the WORKBENCH_API_KEY info
        api_key_info = self.get_api_key_info()
        output["API_KEY_INFO"] = api_key_info
        return output

    def set_config(self, key: str, value: Any):
        """Set a configuration value for the given key.

        Args:
            key (str): The configuration key to set.
            value (Any): The value for the configuration key.
        """
        self.config[key] = value

    def overwrite_config_with_env(self):
        """Overwrite the configuration with environment variables."""
        overwrites = [
            "WORKBENCH_ROLE",
            "WORKBENCH_BUCKET",
            "WORKBENCH_API_KEY",
            "WORKBENCH_PLUGINS",
            "WORKBENCH_THEMES" "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_PASSWORD",
        ]
        for key, value in os.environ.items():
            # If the key is in the overwrites list, then overwrite the config
            if key in overwrites:
                self.log.important(f"Overwriting {key} with ENV var: {value}")
                self.config[key] = value

    def overwrite_config_with_parameter_store(self):
        """Overwrite the configuration with AWS Parameter Store."""
        """FIXME: Need to resolve circular dependency between ConfigManager and ParameterStore"""
        self.log.important("Stubbed out: Overwriting config with AWS Parameter Store...")
        """
        from workbench.api.parameter_store import ParameterStore

        overwrites = [
            "WORKBENCH_ROLE",
            "WORKBENCH_BUCKET",
            "WORKBENCH_API_KEY",
            "WORKBENCH_PLUGINS",
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_PASSWORD"
        ]
        params = ParameterStore()
        for key in overwrites:
            config_key = f"/config/{key}"
            value = params.get(config_key, warn=False)
            if value:
                self.log.important(f"Overwriting {key} with Parameter Store: {value}")
                self.config[key] = value
        """

    def is_open_source(self) -> bool:
        """Returns True if the API is open source."""
        api_key_info = self.get_api_key_info()
        if api_key_info["license_id"] == "Open Source":
            return True
        return False

    def open_source_api_key(self) -> str:
        """Read the open source API key from the package resources.

        Returns:
            str: The open source API key.
        """
        # Python 3.9 compatibility
        with get_resource_path("workbench.resources", "open_source_api.key") as open_source_key_path:
            with open(open_source_key_path, "r") as key_file:
                return key_file.read().strip()

    def create_site_config(self):
        """Create a site configuration file from the default configuration."""
        site_config_updates = {}

        # Grab the bootstrap config
        bootstrap_config = self._load_bootstrap_config()

        # Prompt for each configuration value
        for key, value in bootstrap_config.items():
            if value == "change_me":
                value = input(f"{key}: ")
                site_config_updates[key] = value
            elif "change_me_optional" in value:
                # If the value has a : in it then the part after the : is the default value
                if ":" in value:
                    default_value = value.split(":")[1].strip()
                    value = input(f"[optional] {key}({default_value}): ")
                    if value in ["", None]:
                        site_config_updates[key] = default_value
                    else:
                        site_config_updates[key] = value

                    # Special logic for WORKBENCH_API_KEY
                    if key == "WORKBENCH_API_KEY" and site_config_updates[key] == default_value:
                        print("Using Open Source API Key...")
                        site_config_updates[key] = self.open_source_api_key()
                else:
                    value = input(f"[optional] {key}: ")
                    if value in ["", None]:
                        site_config_updates[key] = None

        # Update default config with provided values
        site_config = {**bootstrap_config, **site_config_updates}

        # Determine platform-specific path (e.g., ~/.workbench/config.json)
        self.site_config_path = self.get_platform_specific_path()

        # Save updated config to platform-specific path
        with open(self.site_config_path, "w") as file:
            json.dump(site_config, file, indent=4)

    def config_okay(self) -> bool:
        """Returns True if the configuration is okay."""
        required_keys = ["WORKBENCH_ROLE", "WORKBENCH_BUCKET", "WORKBENCH_API_KEY"]
        for key in required_keys:
            if key not in self.config:
                self.log.critical(f"Missing required config: {key}")
                return False

        # Also make sure that the WORKBENCH_BUCKET is not the default value
        if self.config["WORKBENCH_BUCKET"] == "env-will-overwrite":
            self.overwrite_config_with_env()
            if self.config["WORKBENCH_BUCKET"] == "env-will-overwrite":
                self.log.critical("WORKBENCH_BUCKET needs to be set with ENV var...")
                return False

        return True

    def get_api_key_info(self) -> Dict[str, Any]:
        """Get the API Key information from the configuration.

        Returns:
            Dict[str, Any]: API Key information.
        """
        api_key = self.get_config("WORKBENCH_API_KEY")
        api_info = self.license_manager.load_api_license(aws_account_id=None, api_key=api_key)
        return api_info

    def get_license_id(self) -> str:
        """Get the license ID from the license information

        Returns:
            str: The license ID
        """
        return self.get_api_key_info().get("license_id", "Unknown")

    def load_and_check_license(self, aws_account_id: int) -> bool:
        """Check the license for expiration and signature verification.

        Args:
            aws_account_id (int): The AWS account ID (for account specific licenses).

        Returns:
            bool: True if the license is okay.
        """
        is_valid = self.license_manager.load_api_license(aws_account_id, self.get_config("WORKBENCH_API_KEY"))
        return is_valid

    def print_license_info(self):
        """Print the license information to the log."""
        self.license_manager.print_license_info()

    @staticmethod
    def get_platform_specific_path() -> str:
        """Returns the platform-specific path for the config file.

        Returns:
            str: Path for the config file.
        """
        home_dir = os.path.expanduser("~")
        config_file_name = "workbench_config.json"

        if platform.system() == "Windows":
            # Use AppData\Local
            config_path = os.path.join(home_dir, "AppData", "Local", "Workbench", config_file_name)
        else:
            # For macOS and Linux, use a hidden file in the home directory
            config_path = os.path.join(home_dir, ".workbench", config_file_name)

        # Ensure the directory exists and return the path
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        return config_path

    def platform_specific_instructions(self):
        """Provides instructions to the user for setting the WORKBENCH_CONFIG
        environment variable permanently based on their operating system.
        """
        os_name = platform.system()

        if os_name == "Windows":
            instructions = (
                "\nTo set the WORKBENCH_CONFIG environment variable permanently on Windows:\n"
                "1. Press Win + R, type 'sysdm.cpl', and press Enter.\n"
                "2. Go to the 'Advanced' tab and click on 'Environment Variables'.\n"
                "3. Under 'System variables', click 'New'.\n"
                "4. Set 'Variable name' to 'WORKBENCH_CONFIG' and 'Variable value' to '{}'.\n"
                "5. Click OK and Apply. You might need to restart your system for changes to take effect."
            ).format(self.site_config_path)

        elif os_name in ["Linux", "Darwin"]:  # Darwin is macOS
            shell_files = {"Linux": "~/.bashrc or ~/.profile", "Darwin": "~/.bash_profile, ~/.zshrc, or ~/.zprofile"}
            instructions = (
                "\nTo set the WORKBENCH_CONFIG environment variable permanently on {}:\n"
                "1. Open {} in a text editor.\n"
                "2. Add the following line at the end of the file:\n"
                "   export WORKBENCH_CONFIG='{}'\n"
                "3. Save the file and restart your terminal for the changes to take effect."
            ).format(os_name, shell_files[os_name], self.site_config_path)

        else:
            instructions = f"OS not recognized. Set the WORKBENCH_CONFIG ENV var to {self.site_config_path} manually."

        print(instructions)

    def _load_config(self) -> Dict[str, Any]:
        """Internal: Load configuration based on the WORKBENCH_CONFIG environment variable.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """

        # Load site_config_path from environment variable
        self.site_config_path = os.environ.get("WORKBENCH_CONFIG")
        if self.site_config_path is None:
            self.log.warning("WORKBENCH_CONFIG ENV var not set")
            return self._load_default_config()

        # Load site specific configuration file
        try:
            # Normalize the path
            self.site_config_path = os.path.normpath(self.site_config_path)
            self.log.info(f"Loading site configuration from {self.site_config_path}...")
            with open(self.site_config_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            self.log.error(f"Config file not found at {self.site_config_path}. Loading default config.")
            return self._load_default_config()
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to decode JSON from {self.site_config_path}: {e}. Loading default config.")
            return self._load_default_config()

    @staticmethod
    def _load_bootstrap_config() -> Dict[str, Any]:
        """Internal: Load the bootstrap configuration from the package resources.

        Returns:
            Dict[str, Any]: Bootstrap configuration dictionary.
        """
        bootstrap_config = {
            "AWS_PROFILE": "change_me",
            "WORKBENCH_BUCKET": "change_me",
            "REDIS_HOST": "change_me_optional:localhost",
            "REDIS_PORT": "change_me_optional:6379",
            "REDIS_PASSWORD": "change_me_optional:",
            "WORKBENCH_ROLE": "Workbench-ExecutionRole",
            "WORKBENCH_PLUGINS": "package",
            "WORKBENCH_FEATURES": {
                "plugins": "true",
                "experimental": "false",
                "large_meta_data": "false",
                "enterprise": "false",
            },
            "WORKBENCH_API_KEY": "change_me_optional:open_source",
        }
        return bootstrap_config

    def _load_default_config(self) -> Dict[str, Any]:
        """Internal: Load default configuration and combine with any existing environment variables.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        self.using_default_config = True
        self.log.warning("Loading default config and pulling ENV vars...")
        config = {
            "WORKBENCH_ROLE": "Workbench-ExecutionRole",
            "WORKBENCH_PLUGINS": "package",
            "WORKBENCH_API_KEY": self.open_source_api_key(),
        }
        for key, value in os.environ.items():
            if key.startswith("WORKBENCH_") or key.startswith("REDIS_") or key == "AWS_PROFILE":
                config[key] = value
        return config


if __name__ == "__main__":
    """Exercise the ConfigManager class"""
    from pprint import pprint

    cm = ConfigManager()
    workbench_role = cm.get_config("WORKBENCH_ROLE")
    print(f"WORKBENCH_ROLE: {workbench_role}")
    workbench_plugins = cm.get_config("WORKBENCH_PLUGINS")
    print(f"WORKBENCH_PLUGINS: {workbench_plugins}")

    # License ID
    print(f"WORKBENCH_LICENSE_ID: {cm.get_license_id()}")

    # API Key Info
    my_api_key_info = cm.get_api_key_info()
    pprint(my_api_key_info)

    # All config
    pprint(cm.get_all_config())

    # Unset WORKBENCH_CONFIG
    os.environ.pop("WORKBENCH_CONFIG", None)
    ConfigManager._instance = None  # We need to reset the singleton instance for testing

    # Add the WORKBENCH_BUCKET and REDIS_HOST to the ENV vars
    os.environ["WORKBENCH_BUCKET"] = "bucket-from-env"
    cm = ConfigManager()
    pprint(cm.get_all_config())

    # Simulate running on Docker
    def running_on_docker() -> bool:  # noqa: F811
        return True

    ConfigManager._instance = None  # We need to reset the singleton instance for testing
    os.environ.pop("WORKBENCH_BUCKET", None)
    cm = ConfigManager()
    cm.set_config("WORKBENCH_BUCKET", "bucket-from-set_config")
    pprint(cm.get_all_config())

    # Test set_config()
    cm.set_config("WORKBENCH_BUCKET", "bucket-from-set_config")
    cm_new = ConfigManager()
    pprint(cm_new.get_all_config())

    # Test ENV var overwrite
    os.environ["WORKBENCH_BUCKET"] = "bucket-from-env"
    os.environ["REDIS_HOST"] = "localhost"
    cm = ConfigManager()
    cm.overwrite_config_with_env()
    pprint(cm.get_all_config())

    # Test not having enough config
    ConfigManager._instance = None  # We need to reset the singleton instance for testing
    os.environ.pop("WORKBENCH_BUCKET", None)

    # This will fail with a FatalConfigError (which is good)
    cm = ConfigManager()
