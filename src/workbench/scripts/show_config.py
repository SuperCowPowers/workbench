"""Show the current Workbench Config"""

from workbench.utils.config_manager import ConfigManager
from workbench.utils.repl_utils import cprint


# Show the current Workbench Config
def main():
    # Create a ConfigManager instance
    cm = ConfigManager()

    # Show the current Workbench Config
    cprint("yellow", f"\n\nWorkbench Config Path: {cm.site_config_path}")
    config = cm.get_all_config()
    for key, value in config.items():
        cprint(["lightpurple", "\t" + key, "lightgreen", value])


if __name__ == "__main__":
    main()
