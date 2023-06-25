"""Configuration Utilities"""
import os
import sys
from pathlib import Path
from configparser import ConfigParser
import logging

# SageWorks Imports
from sageworks.utils.sageworks_logging import logging_setup


# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


class SageWorksConfig:
    """SageWorksConfig provides a set of utilities to read the SageWorks config file"""

    def __init__(self):
        self.log = logging.getLogger(__file__)

        # Locate the configuration file
        self.config_file = self.locate_config_file()
        if not self.config_file:
            self.log.critical("Unable to locate SageWorks Config File...")
            sys.exit(1)

        # Read in the configuration file
        self.sageworks_config = ConfigParser()
        self.sageworks_config.read(self.config_file)

    def locate_config_file(self):
        """Locate the sageworks configuration file"""

        # Check for an ENV Var for the config file path
        env_config_file = os.environ.get("SAGEWORKS_CONFIG_FILE")
        if env_config_file:
            self.log.info(f"Using ENV VAR for SAGEWORKS_CONFIG_FILE: {env_config_file}")

            # Check for configuration file existence
            if os.path.exists(env_config_file):
                return env_config_file
            else:
                self.log.warning(f"Config Not Found: {env_config_file} does not exist")

        # Check the User's home directory for a config file
        home_config_file = Path.home() / ".config" / "sageworks" / "sageworks_config.ini"
        if os.path.exists(home_config_file):
            return home_config_file
        else:
            self.log.warning(f"Config Not Found: {home_config_file} does not exist")

        # Last resort, check the git repository for a config file
        repo_config_file = self.get_repository_config_path() / "config" / "sageworks_config.ini"
        if os.path.exists(repo_config_file):
            self.log.info("Using Repository Config... you probably want to fix this")
            self.log.info(f"Repository Config: {repo_config_file}")
            return repo_config_file
        else:
            self.log.warning(f"Config Not Found: {repo_config_file} does not exist")

        # Totally failed to find a config file
        return None

    @staticmethod
    def get_repository_config_path():
        """Get the default config path for finding the config from the git repository"""
        return Path(sys.modules["sageworks"].__file__).parent.parent.parent

    def get_config_sections(self) -> list:
        """Grab the section names out of the config file"""
        return self.sageworks_config.sections()

    def get_config_section_keys(self, section: str) -> list:
        """Grab the section names out of the config file"""
        return list(self.sageworks_config[section])

    def get_config_value(self, section: str, key: str):
        """Get a specific value from the config file"""
        try:
            return self.sageworks_config[section][key]
        except KeyError:
            log.critical("Could not find config key: {:s}:{:s}".format(section, key))
            return None


if __name__ == "__main__":
    """Exercise the SageWorks config utility methods"""

    # Create a SageWorksConfig object
    sw_config = SageWorksConfig()

    config_sections = sw_config.get_config_sections()
    for section in config_sections:
        print(f"Config Section: {section}")
        keys = sw_config.get_config_section_keys(section)
        for key in keys:
            print(f"\t{key}: {sw_config.get_config_value(section, key)}")
