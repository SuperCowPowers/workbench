# Configuration Manager Class (Singleton Pattern)
import json
import os
import platform
import logging
from typing import Any, Dict

# Workbench imports
from workbench.utils.execution_environment import running_as_service


class FatalConfigError(Exception):
    """Exception raised for errors in the configuration."""


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

        # Check if we're running as a service
        self.running_as_service = False
        if running_as_service():
            self.log.monitor("Running as part of a Service...")
            self.running_as_service = True

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
        # Special logic for WORKBENCH_PLUGINS: "package" means the plugin pages that ship
        # with Workbench, so a dashboard has something to show without any configuration.
        if key == "WORKBENCH_PLUGINS":
            plugin_dir = self.config.get(key, default_value)
            if plugin_dir in ["package", "", None]:
                return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "plugin_pages"))
            else:
                return plugin_dir

        # Special logic for WORKBENCH_LOCAL_PATH: storage root for workbench.local artifacts
        if key == "WORKBENCH_LOCAL_PATH":
            local_path = self.config.get(key)
            if local_path in ["", None]:
                return os.path.join(os.path.expanduser("~"), ".workbench", "local")
            return os.path.expanduser(local_path)

        # Special logic for ML_PIPELINES_ROOT (S3 prefix or local directory)
        if key == "ML_PIPELINES_ROOT":
            pipelines_root = self.config.get(key)
            if pipelines_root in ["", None]:
                bucket = self.config.get("WORKBENCH_BUCKET")
                return f"s3://{bucket}/ml_pipelines" if bucket else default_value
            return pipelines_root

        # Normal logic
        return self.config.get(key, default_value)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dict[str, Any]: All configuration values.
        """
        output = dict(self.config)
        output["UI_UPDATE_RATE"] = self.ui_update_rate()
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
            "WORKBENCH_PLUGINS",
            "WORKBENCH_THEMES",
            "ML_PIPELINES_ROOT",
            "WORKBENCH_LOCAL_PATH",
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_PASSWORD",
            "DASHBOARD_URL",
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

    def ui_update_rate(self) -> int:
        """Get the UI update rate from the configuration.

        Returns:
            int: The UI update rate in seconds.
        """
        # Default to 60 seconds if not set
        return int(self.config.get("UI_UPDATE_RATE", 60))

    def create_site_config(self):
        """Create a site configuration file from the default configuration."""
        site_config_updates = {}

        # Grab the bootstrap config
        bootstrap_config = self._load_bootstrap_config()

        # Prompt for each configuration value
        for key, value in bootstrap_config.items():
            if key == "ENABLE_BOSCO":
                answer = input("[optional] ENABLE_BOSCO -- run the Bosco ML agent? (y/N): ").strip().lower()
                site_config_updates[key] = answer in ("y", "yes", "true", "1")
            elif not isinstance(value, str):
                # Non-string defaults (dicts, bools) are kept as-is, not prompted
                continue
            elif value == "change_me":
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
        # With no site config the user is in local mode, which is a supported state.
        # Only an incomplete site config -- they meant to connect -- is alarming.
        report = self.log.info if self.using_default_config else self.log.critical

        required_keys = ["WORKBENCH_ROLE", "WORKBENCH_BUCKET"]
        for key in required_keys:
            if key not in self.config:
                report(f"Missing required config: {key}")
                return False

        # Also make sure that the WORKBENCH_BUCKET is not the default value
        if self.config["WORKBENCH_BUCKET"] == "env-will-overwrite":
            self.overwrite_config_with_env()
            if self.config["WORKBENCH_BUCKET"] == "env-will-overwrite":
                report("WORKBENCH_BUCKET needs to be set with ENV var...")
                return False

        return True

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
        if self.site_config_path is None or self.site_config_path == "":
            self.log.debug("WORKBENCH_CONFIG ENV var not set")
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
            "DASHBOARD_URL": "change_me_optional:",
            "WORKBENCH_ROLE": "Workbench-BuilderRole",
            "WORKBENCH_PLUGINS": "package",
            "ENABLE_BOSCO": False,
            "WORKBENCH_FEATURES": {
                "plugins": "true",
                "experimental": "false",
                "large_meta_data": "false",
                "enterprise": "false",
            },
        }
        return bootstrap_config

    def _load_default_config(self) -> Dict[str, Any]:
        """Internal: Load default configuration and combine with any existing environment variables.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        self.using_default_config = True
        self.log.debug("Loading default config and pulling ENV vars...")
        config = {
            "WORKBENCH_ROLE": "Workbench-ExecutionRole",
            "WORKBENCH_PLUGINS": "package",
        }
        for key, value in os.environ.items():
            if key.startswith(("WORKBENCH_", "REDIS_")) or key in ["AWS_PROFILE", "ML_PIPELINES_ROOT"]:
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

    # All config
    pprint(cm.get_all_config())

    # Get the UI update rate
    ui_update_rate = cm.ui_update_rate()
    print(f"UI_UPDATE_RATE: {ui_update_rate}")

    # Unset WORKBENCH_CONFIG
    os.environ.pop("WORKBENCH_CONFIG", None)
    ConfigManager._instance = None  # We need to reset the singleton instance for testing

    # Add the WORKBENCH_BUCKET and REDIS_HOST to the ENV vars
    os.environ["WORKBENCH_BUCKET"] = "bucket-from-env"
    cm = ConfigManager()
    pprint(cm.get_all_config())

    # Simulate running as a service
    def running_as_service() -> bool:  # noqa: F811
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
    ConfigManager()
