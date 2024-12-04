import json
import os
import logging
from pathlib import Path
import plotly.io as pio
import dash_bootstrap_components as dbc
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
        self.current_theme = 'dark'  # Default theme
        self.dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

        # Load available themes
        self.load_themes()

        # Set the default theme
        self.set_theme(default_theme)

    def load_themes(self):
        """
        Load available themes from the themes directory.
        Each theme is expected to have a subdirectory containing:
        - JSON for Plotly template (e.g., `dark_template.json`).
        - CSS files (e.g., `base.css`, `custom.css`, `tweaks.css`).
        """
        if not self.themes_dir:
            self.log.warning("No themes directory specified.")
            return

        for theme_dir in self.themes_dir.iterdir():
            if theme_dir.is_dir():
                theme_name = theme_dir.name
                plotly_template = theme_dir / f"{theme_name}_template.json"
                css_files = [
                    theme_dir / "base.css",
                    theme_dir / "custom.css",
                    theme_dir / "tweaks.css",
                ]

                # Check for the required files
                if not plotly_template.exists():
                    self.log.warning(f"Missing Plotly template for theme '{theme_name}'")
                    plotly_template = None

                # Add theme to available themes
                self.available_themes[theme_name] = {
                    "plotly_template": plotly_template,
                    "css_files": [css_file for css_file in css_files if css_file.exists()],
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
        with open(theme["plotly_template"], "r") as f:
            template = json.load(f)
        pio.templates["custom_template"] = template
        pio.templates.default = "custom_template"

        # Dynamically load or switch the CSS
        self._reload_css(theme["css_files"])

        # Update the current theme
        self.current_theme = theme_name

    def get_current_theme(self) -> str:
        """
        Get the name of the current theme.

        Returns:
            str: The name of the current theme.
        """
        return self.current_theme

    def get_current_css_files(self) -> list[str]:
        """
        Get the list of CSS files for the current theme.

        Returns:
            list[str]: List of CSS files for the current theme.
        """

        theme = self.available_themes[self.current_theme]
        theme_css = theme["css_files"]
        base_css = [dbc.themes.DARKLY, self.dbc_css]
        return base_css + [str(css_file) for css_file in theme_css]

    def _reload_css(self, css_files: list[Path]):
        """
        Reload or switch the application's CSS.

        Args:
            css_files (list[Path]): List of CSS files to load.

        TODO: Implement dynamic CSS reloading in the app.
        """
        for css_file in css_files:
            # For now, just log the CSS path to simulate loading
            print(f"Reloading CSS from {css_file}")

if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager(default_theme="light")

    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.get_current_theme())

    # Set a new theme
    theme_manager.set_theme("dark")
    print("Theme switched to:", theme_manager.get_current_theme())

    print("Current Theme:", theme_manager.get_current_theme())
    print("CSS Files for Current Theme:", theme_manager.get_current_css_files())
