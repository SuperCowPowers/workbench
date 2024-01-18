import json
import os
import platform
import logging
import importlib.resources as resources
from typing import Any, Dict

# SageWorks imports
from sageworks.utils.license_manager import LicenseManager


class ConfigManager:
    def __init__(self, interactive: bool = False):
        """Initialize the ConfigManager

        Args:
            interactive (bool, optional): If True, prompt the user for configuration values. Defaults to False.
        """
        self.log = logging.getLogger("sageworks")
        self.interactive = interactive
        self.site_config_path = None
        self.needs_bootstrap = False
        self.config = self._load_config()

    def get_config(self, key: str) -> Any:
        """Get a configuration value by key.

        Args:
            key (str): The configuration key to retrieve.

        Returns:
            Any: The value of the configuration key.
        """
        # Special logic for SAGEWORKS_PLUGINS
        if key == "SAGEWORKS_PLUGINS":
            plugin_dir = self.config.get(key, None)
            if plugin_dir in ["package", "", None]:
                return os.path.join(os.path.dirname(__file__), "../../../applications/aws_dashboard/sageworks_plugins")
            else:
                return self.config.get(key, None)

        # Normal logic
        return self.config.get(key, None)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dict[str, Any]: All configuration values.
        """
        # Grab all config except the SAGEWORKS_API_KEY
        output = {key: value for key, value in self.config.items() if key != "SAGEWORKS_API_KEY"}

        # Add the SAGEWORKS_API_KEY info
        api_key_info = self.get_api_key_info()
        output["API_KEY_INFO"] = api_key_info
        return output

    def is_open_source(self) -> bool:
        """Returns True if the API is open source."""
        api_key_info = self.get_api_key_info()
        if api_key_info["license_id"] == "Open Source":
            return True
        return False

    @staticmethod
    def open_source_api_key() -> str:
        """Read the open source API key from the package resources.

        Returns:
            str: The open source API key.
        """
        with resources.path("sageworks.resources", "open_source_api.key") as open_source_key_path:
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

                    # Special logic for SAGEWORKS_API_KEY
                    if key == "SAGEWORKS_API_KEY" and site_config_updates[key] == default_value:
                        print("Using Open Source API Key...")
                        site_config_updates[key] = self.open_source_api_key()
                else:
                    value = input(f"[optional] {key}: ")
                    if value in ["", None]:
                        site_config_updates[key] = None

        # Update default config with provided values
        site_config = {**bootstrap_config, **site_config_updates}

        # Determine platform-specific path (e.g., ~/.sageworks/config.json)
        self.site_config_path = self.get_platform_specific_path()

        # Save updated config to platform-specific path
        with open(self.site_config_path, "w") as file:
            json.dump(site_config, file, indent=4)

        # Done with bootstrap
        self.needs_bootstrap = False

    def _load_config(self) -> Dict[str, Any]:
        """Internal: Load configuration based on the SAGEWORKS_CONFIG environment variable.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """

        # Load site_config_path from environment variable
        self.site_config_path = os.environ.get("SAGEWORKS_CONFIG")
        if self.site_config_path is None:
            if self.interactive:
                self.log.warning("SAGEWORKS_CONFIG ENV var not set. [Interactive] bootstrap configuration...")
                return self._load_bootstrap_config()
            else:
                self.log.warning("SAGEWORKS_CONFIG ENV var not set. Loading Default config...")
                return self._load_default_config()

        # Load configuration from AWS Parameter Store
        if self.site_config_path == "parameter_store":
            try:
                self.log.info("Loading site configuration from AWS Parameter Store...")
                return self.get_config_from_aws_parameter_store()
            except Exception:
                self.log.error("Failed to load config from AWS Parameter Store. Loading Default config...")
                return self._load_default_config()

        # Load site specified configuration file
        try:
            self.log.info(f"Loading site configuration from {self.site_config_path}...")
            with open(self.site_config_path, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.log.error(f"Failed to load config from {self.site_config_path}. Loading Default config...")
            return self._load_default_config()

    def _load_bootstrap_config(self) -> Dict[str, Any]:
        """Internal: Load the bootstrap configuration from the package resources.

        Returns:
            Dict[str, Any]: Bootstrap configuration dictionary.
        """
        self.needs_bootstrap = True
        with resources.open_text("sageworks.resources", "bootstrap_config.json") as file:
            return json.load(file)

    def _load_default_config(self) -> Dict[str, Any]:
        """Internal: Load default configuration and combine with any existing environment variables.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        config = {
            "SAGEWORKS_ROLE": "SageWorks-ExecutionRole",
            "SAGEWORKS_PLUGINS": "package",
            "SAGEWORKS_API_KEY": self.open_source_api_key(),
        }
        for key, value in os.environ.items():
            if key.startswith("SAGEWORKS_") or key.startswith("REDIS_"):
                config[key] = value
        return config

    def get_api_key_info(self) -> Dict[str, Any]:
        """Get the API Key information from the configuration.

        Returns:
            Dict[str, Any]: API Key information.
        """
        api_key = self.get_config("SAGEWORKS_API_KEY")
        api_info = LicenseManager().load_api_license(aws_account_id=None, api_key=api_key)
        return api_info

    @staticmethod
    def get_platform_specific_path() -> str:
        """Returns the platform-specific path for the config file.

        Returns:
            str: Path for the config file.
        """
        home_dir = os.path.expanduser("~")
        config_file_name = "sageworks_config.json"

        if platform.system() == "Windows":
            # Use AppData\Local
            config_path = os.path.join(home_dir, "AppData", "Local", "SageWorks", config_file_name)
        else:
            # For macOS and Linux, use a hidden file in the home directory
            config_path = os.path.join(home_dir, ".sageworks", config_file_name)

        # Ensure the directory exists and return the path
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        return config_path

    def get_config_from_aws_parameter_store(self) -> Dict[str, Any]:
        """Stub method to fetch configuration from AWS Parameter Store.

        Returns:
            Dict[str, Any]: Configuration dictionary from AWS Parameter Store.
        """
        # TODO: Implement AWS Parameter Store fetching logic
        return {}

    def platform_specific_instructions(self):
        """Provides instructions to the user for setting the SAGEWORKS_CONFIG
        environment variable permanently based on their operating system.
        """
        os_name = platform.system()

        if os_name == "Windows":
            instructions = (
                "\nTo set the SAGEWORKS_CONFIG environment variable permanently on Windows:\n"
                "1. Press Win + R, type 'sysdm.cpl', and press Enter.\n"
                "2. Go to the 'Advanced' tab and click on 'Environment Variables'.\n"
                "3. Under 'System variables', click 'New'.\n"
                "4. Set 'Variable name' to 'SAGEWORKS_CONFIG' and 'Variable value' to '{}'.\n"
                "5. Click OK and Apply. You might need to restart your system for changes to take effect."
            ).format(self.site_config_path)

        elif os_name in ["Linux", "Darwin"]:  # Darwin is macOS
            shell_files = {"Linux": "~/.bashrc or ~/.profile", "Darwin": "~/.bash_profile, ~/.zshrc, or ~/.profile"}
            instructions = (
                "\nTo set the SAGEWORKS_CONFIG environment variable permanently on {}:\n"
                "1. Open {} in a text editor.\n"
                "2. Add the following line at the end of the file:\n"
                "   export SAGEWORKS_CONFIG='{}'\n"
                "3. Save the file and restart your terminal for the changes to take effect."
            ).format(os_name, shell_files[os_name], self.site_config_path)

        else:
            instructions = f"OS not recognized. Set the SAGEWORKS_CONFIG ENV var to {self.site_config_path} manually."

        print(instructions)


if __name__ == "__main__":
    """Exercise the ConfigManager class"""
    from pprint import pprint

    cm = ConfigManager()
    sageworks_role = cm.get_config("SAGEWORKS_ROLE")
    print(f"SAGEWORKS_ROLE: {sageworks_role}")
    sageworks_plugins = cm.get_config("SAGEWORKS_PLUGINS")
    print(f"SAGEWORKS_PLUGINS: {sageworks_plugins}")

    # API Key Info
    my_api_key_info = cm.get_api_key_info()
    pprint(my_api_key_info)

    # All config
    pprint(cm.get_all_config())

    # Unset SAGEWORKS_CONFIG
    os.environ.pop("SAGEWORKS_CONFIG", None)

    # Add the SAGEWORKS_BUCKET and REDIS_HOST to the ENV vars
    os.environ["SAGEWORKS_BUCKET"] = "ideaya-sageworks-bucket"
    os.environ["REDIS_HOST"] = "sageworksredis.qo8vb5.0001.use1.cache.amazonaws.com"
    cm = ConfigManager()
    pprint(cm.get_all_config())
