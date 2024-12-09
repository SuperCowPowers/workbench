"""Show the current SageWorks Config"""

from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.repl_utils import cprint


# Show the current SageWorks Config
def main():
    # Create a ConfigManager instance
    cm = ConfigManager()

    # Show the current SageWorks Config
    cprint("yellow", f"\n\nSageWorks Config Path: {cm.site_config_path}")
    config = cm.get_all_config()
    for key, value in config.items():
        cprint(["lightpurple", "\t" + key, "lightgreen", value])


if __name__ == "__main__":
    main()
