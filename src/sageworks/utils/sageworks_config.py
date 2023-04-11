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

        # FIXME: This is a hack to get the config file path
        self.default_config_path = self.get_default_config_path()
        self.config_file = self.get_config_file_path()

        # Read in the configuration file
        self.sageworks_config = ConfigParser()
        self.sageworks_config.read(self.config_file)

    @staticmethod
    def get_default_config_path():
        """Get the default config path"""
        # FIXME: This is a hack to get the config file path
        return Path(sys.modules["sageworks"].__file__).parent.parent.parent

    def get_config_file_path(self):
        """Find the config file"""

        # Do we have an ENV var for the test_data path or config file?
        config_path = os.environ.get('SAGEWORKS_CONFIG_PATH') or self.default_config_path
        config_file = os.environ.get('SAGEWORKS_CONFIG_FILE') or 'sageworks_config.ini'
        return os.path.join(config_path, config_file)

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
            log.critical('Could not find config key: {:s}:{:s}'.format(section, key))
            return None


if __name__ == '__main__':
    """Exercise the SageWorks config utility methods"""

    # Create a SageWorksConfig object
    sw_config = SageWorksConfig()

    config_sections = sw_config.get_config_sections()
    print(config_sections)
    for section in config_sections:
        keys = sw_config.get_config_section_keys(section)
        for key in keys:
            print(key, sw_config.get_config_value(section, key))

    # Try an unknown config key
    sw_config.get_config_value('UNKNOWN', 'KEY')
