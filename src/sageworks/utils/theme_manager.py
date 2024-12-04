import json
import os
import logging
from pathlib import Path
import plotly.io as pio
from sageworks.utils.config_manager import ConfigManager


class ThemeManager:
    """
    A class to manage themes (Plotly templates and CSS) for a Dash application.
    """

    def __init__(self, default_theme: str = "light"):
        """
        Initialize the ThemeManager with a directory containing themes.

        Args:
            default_theme (str): Default theme to load (e.g., 'light' or 'dark').
        """
        self.log = logging.getLogger("sageworks")

        # Get themes directory from the configuration manager
        cm = ConfigManager()
        theme_path = cm.get_config("SAGEWORKS_THEMES")
        self.themes_dir = Path(theme_path) if theme_path else None
        if self.themes_dir is None or not self.themes_dir.exists():
            self.log.error(f"The themes directory '{self.themes_dir}' does not exist.")

        self.available_themes = {}
        self.current_theme = None

        # Load available themes
        self.load_themes()

        # Set the default theme
        self.set_theme(default_theme)

    def load_themes(self):
        """
        Load available themes from the themes directory.
        Looks for JSON files (component templates) and CSS files.
        """
        if not self.themes_dir:
            self.log.important("No themes directory specified.")
            return
        for theme_file in self.themes_dir.glob("*.json"):
            theme_name = theme_file.stem
            self.available_themes[theme_name] = {
                "component_template": theme_file,
                "css": self.themes_dir / f"{theme_name}.css",
            }

        if not self.available_themes:
            self.log.warning(f"No themes found in '{self.themes_dir}'.")

    def list_themes(self) -> list:
        """
        List all available themes.

        Returns:
            list: A list of theme names.
        """
        return list(self.available_themes.keys())

    def set_theme(self, theme_name: str):
        """
        Set the application's theme.

        Args:
            theme_name (str): The name of the theme to set (e.g., 'light' or 'dark').

        Raises:
            ValueError: If the theme is not available.
        """
        if theme_name not in self.available_themes:
            self.log.error(f"Theme '{theme_name}' is not available.")
            return

        theme = self.available_themes[theme_name]

        # Set Plotly template
        with open(theme["component_template"], "r") as f:
            template = json.load(f)
        pio.templates["custom_template"] = template
        pio.templates.default = "custom_template"

        # Dynamically load or switch the CSS
        self._reload_css(theme["css"])

        # Update the current theme
        self.current_theme = theme_name

    def _reload_css(self, css_file: Path):
        """
        Reload or switch the application's CSS.

        Args:
            css_file (Path): Path to the CSS file to load.

        TODO: Implement dynamic CSS reloading in the app.
        """
        # For now, print the CSS path to simulate loading
        print(f"Reloading CSS from {css_file}")

    def get_current_theme(self):
        """
        Get the name of the current theme.

        Returns:
            str: The name of the current theme.
        """
        return self.current_theme


if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager(default_theme="light")

    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.get_current_theme())

    # Set a new theme
    theme_manager.set_theme("dark")
    print("Theme switched to:", theme_manager.get_current_theme())
