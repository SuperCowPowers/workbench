import json
import logging
from pathlib import Path
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory
from sageworks.utils.config_manager import ConfigManager


class ThemeManager:
    _instance = None  # Singleton instance

    # Class-level state
    _log = logging.getLogger("sageworks")
    _themes_dir = None
    _bootstrap_themes = {
        "dark": dbc.themes.DARKLY,
        "light": dbc.themes.FLATLY,
        "minty": dbc.themes.MINTY,
        "quartz": dbc.themes.QUARTZ,
    }
    _dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    _available_themes = {}
    _current_theme_name = None
    _current_template = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize()  # Initialize class-level state
        return cls._instance

    @classmethod
    def _initialize(cls):
        """Initialize the ThemeManager's shared state."""
        cm = ConfigManager()
        theme_path = cm.get_config("SAGEWORKS_THEMES")
        cls._themes_dir = Path(theme_path) if theme_path else None

        if not cls._themes_dir or not cls._themes_dir.exists():
            cls._log.error(f"The themes directory '{cls._themes_dir}' does not exist.")

        cls._load_themes()
        cls.set_theme("dark")  # Default theme

    @classmethod
    def _load_themes(cls):
        """Load available themes."""
        if not cls._themes_dir:
            cls._log.warning("No themes directory specified.")
            return

        for theme_dir in cls._themes_dir.iterdir():
            if theme_dir.is_dir():
                theme_name = theme_dir.name

                # Find the first JSON file in the directory
                json_files = list(theme_dir.glob("*.json"))
                plotly_template = json_files[0] if json_files else None

                # Check for a custom.css file
                custom_css = theme_dir / "custom.css"

                cls._available_themes[theme_name] = {
                    "plotly_template": plotly_template,
                    "custom_css": custom_css if custom_css.exists() else None,
                }

        if not cls._available_themes:
            cls._log.warning(f"No themes found in '{cls._themes_dir}'.")

    @classmethod
    def list_themes(cls) -> list[str]:
        """List available themes."""
        return list(cls._available_themes.keys())

    @classmethod
    def set_theme(cls, theme_name: str):
        """Set the current theme."""
        if theme_name not in cls._available_themes:
            cls._log.error(f"Theme '{theme_name}' is not available.")
            return

        theme = cls._available_themes[theme_name]

        # Update Plotly template
        if theme["plotly_template"]:
            with open(theme["plotly_template"], "r") as f:
                cls._current_template = json.load(f)
            pio.templates["custom_template"] = cls._current_template
            pio.templates.default = "custom_template"
        else:
            cls._log.error(f"No Plotly template found for '{theme_name}'.")

        # Update the current theme name
        cls._current_theme_name = theme_name
        cls._log.important(f"Theme set to '{theme_name}'")

    @classmethod
    def data_bs_theme(cls) -> str:
        """Get the current Bootstrap `data-bs-theme` value."""
        return "dark" if "dark" in cls._current_theme_name.lower() else "light"

    @classmethod
    def current_theme(cls) -> str:
        """Get the name of the current theme."""
        return cls._current_theme_name

    @classmethod
    def colorscale(cls, scale_type: str = "diverging") -> list[list[float | str]]:
        """Get the colorscale for the current theme."""

        # We have 3 colorscale options (diverging, sequential, and sequentialminus)
        color_scales = cls._current_template["layout"]["colorscale"]
        if scale_type in color_scales:
            return color_scales[scale_type]
        else:
            # Use the default colorscale (first one in the dictionary)
            try:
                first_colorscale_name = list(color_scales.keys())[0]
                backup_colorscale = color_scales[first_colorscale_name]
                cls._log.warning(f"Color scale '{scale_type}' not found for template '{cls._current_theme_name}'.")
                cls._log.warning(f"Using color scale '{backup_colorscale}' instead.")
                return backup_colorscale
            except IndexError:
                cls._log.error(f"No color scales found for template '{cls._current_theme_name}'.")
                return []

    @classmethod
    def css_files(cls) -> list[str]:
        """Get the list of CSS files for the current theme."""
        base_css = [cls._bootstrap_themes[cls._current_theme_name], cls._dbc_css]
        theme = cls._available_themes[cls._current_theme_name]
        custom_css = ["/custom.css"] if theme["custom_css"] else []
        return base_css + custom_css

    @classmethod
    def register_css_route(cls, app):
        """Register Flask route for custom.css."""

        @app.server.route("/custom.css")
        def serve_custom_css():
            theme = cls._available_themes[cls._current_theme_name]
            if theme["custom_css"]:
                return send_from_directory(theme["custom_css"].parent, theme["custom_css"].name)
            return "", 404


if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager(theme="dark")
    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())

    theme_manager.set_theme("light")
    print("Theme switched to:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())

    # Example usage of the ThemeManager
    theme_manager = ThemeManager(theme="dark")
    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())
    print("Bootstrap Theme:", theme_manager.data_bs_theme())
    print("Colorscale:", theme_manager.colorscale())
