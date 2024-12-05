import json
import logging
from pathlib import Path
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory
from sageworks.utils.config_manager import ConfigManager


class ThemeManager:
    """
    A class to manage themes (Plotly templates and CSS) for a Dash application.
    """

    def __init__(self, theme: str = "dark"):
        self.log = logging.getLogger("sageworks")
        cm = ConfigManager()
        theme_path = cm.get_config("SAGEWORKS_THEMES")
        self.themes_dir = Path(theme_path) if theme_path else None

        if not self.themes_dir or not self.themes_dir.exists():
            self.log.error(f"The themes directory '{self.themes_dir}' does not exist.")

        self.bootstrap_themes = {
            "dark": dbc.themes.DARKLY,
            "light": dbc.themes.FLATLY,
            "minty": dbc.themes.MINTY,
            "minty_dark": dbc.themes.MINTY,
        }
        self.dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
        self.available_themes = {}
        self.current_theme = None

        self.load_themes()
        self.set_theme(theme)

    def load_themes(self):
        """Load available themes."""
        if not self.themes_dir:
            self.log.warning("No themes directory specified.")
            return

        for theme_dir in self.themes_dir.iterdir():
            if theme_dir.is_dir():
                theme_name = theme_dir.name
                plotly_template = theme_dir / f"{theme_name}_template.json"
                custom_css = theme_dir / "custom.css"

                self.available_themes[theme_name] = {
                    "plotly_template": plotly_template if plotly_template.exists() else None,
                    "custom_css": custom_css if custom_css.exists() else None,
                }

        if not self.available_themes:
            self.log.warning(f"No themes found in '{self.themes_dir}'.")

    def list_themes(self) -> list[str]:
        """List available themes."""
        return list(self.available_themes.keys())

    def set_theme(self, theme_name: str):
        if theme_name not in self.available_themes:
            self.log.error(f"Theme '{theme_name}' is not available.")
            return

        theme = self.available_themes[theme_name]

        # Update Plotly template
        if theme["plotly_template"]:
            with open(theme["plotly_template"], "r") as f:
                template = json.load(f)
            pio.templates["custom_template"] = template
            pio.templates.default = "custom_template"

        # Store the theme name in `current_theme`
        self.current_theme = theme_name
        self.log.info(f"Theme set to '{theme_name}'")

    def get_data_bs_theme(self) -> str:
        """
        Get the current Bootstrap `data-bs-theme` value.

        Returns:
            str: "dark" or "light" based on the current theme.
        """
        return "dark" if "dark" in self.current_theme.lower() else "light"

    def get_current_theme(self) -> str:
        """Get the name of the current theme."""
        return self.current_theme

    def get_current_css_files(self) -> list[str]:
        """
        Get the list of CSS files for the current theme.

        Returns:
            list[str]: List of CSS files for the current theme.
        """
        # Bootstrap CDN and dbc.min.css
        base_css = [self.bootstrap_themes[self.current_theme], self.dbc_css]

        # Use Flask route for custom.css if it exists
        theme = self.available_themes[self.current_theme]
        custom_css = ["/custom.css"] if theme["custom_css"] else []
        return base_css + custom_css

    def get_bs_theme(self) -> str:
        """Get the Bootstrap `data-bs-theme` attribute."""
        return self.current_theme

    def register_css_route(self, app):
        """Register Flask route for custom.css."""
        @app.server.route("/custom.css")
        def serve_custom_css():
            theme = self.available_themes[self.current_theme]
            if theme["custom_css"]:
                return send_from_directory(theme["custom_css"].parent, theme["custom_css"].name)
            return "", 404


if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager(theme="dark")
    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.get_current_theme())
    print("CSS Files for Current Theme:", theme_manager.get_current_css_files())

    theme_manager.set_theme("light")
    print("Theme switched to:", theme_manager.get_current_theme())
    print("CSS Files for Current Theme:", theme_manager.get_current_css_files())