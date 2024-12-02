import json
from pathlib import Path
import plotly.io as pio


class ThemeManager:
    """
    A class to manage themes for the application, including Plotly templates
    and CSS.
    """

    def __init__(self, themes_dir: str, default_theme: str = "light"):
        """
        Initialize the ThemeManager with a directory containing themes.

        Args:
            themes_dir (str): Path to the directory containing theme assets (JSON and CSS).
            default_theme (str): Default theme to load (e.g., 'light' or 'dark').
        """
        self.themes_dir = Path(themes_dir)
        self.available_themes = {}
        self.current_theme = None

        # Load available themes
        self.load_themes()
        # Set the default theme
        self.set_theme(default_theme)

    def load_themes(self):
        """
        Load available themes from the themes directory.
        Looks for JSON files (Plotly templates) and CSS files.
        """
        for theme_file in self.themes_dir.glob("*.json"):
            theme_name = theme_file.stem
            self.available_themes[theme_name] = {
                "plotly_template": theme_file,
                "css": self.themes_dir / f"{theme_name}.css",
            }

    def list_themes(self):
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
            raise ValueError(f"Theme '{theme_name}' is not available.")

        theme = self.available_themes[theme_name]

        # Set Plotly template
        with open(theme["plotly_template"], "r") as f:
            template = json.load(f)
        pio.templates["custom_template"] = template
        pio.templates.default = "custom_template"

        # TBD: Dynamically load or switch the CSS
        # Replace this with actual implementation for injecting CSS dynamically
        self._reload_css(theme["css"])

        # Update the current theme
        self.current_theme = theme_name

    def _reload_css(self, css_file: Path):
        """
        Reload or switch the application's CSS.

        Args:
            css_file (Path): Path to the CSS file to load.

        TBD: Actual implementation for dynamic CSS loading.
        """
        print(f"TBD: CSS reload for {css_file}")  # Placeholder

    def get_current_theme(self):
        """
        Get the name of the current theme.

        Returns:
            str: The name of the current theme.
        """
        return self.current_theme


if __name__ == "__main__":
    # Example usage
    themes_dir = "path/to/themes"  # Replace with your themes directory path
    theme_manager = ThemeManager(themes_dir)

    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.get_current_theme())

    # Set a new theme
    theme_manager.set_theme("dark")
    print("Theme switched to:", theme_manager.get_current_theme())
