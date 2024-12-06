import json
import logging
from pathlib import Path
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager


class ThemeManager:
    _instance = None  # Singleton instance

    # Class-level state
    _log = logging.getLogger("sageworks")
    _themes_dir = None
    _dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    _available_themes = {}
    _current_theme_name = None
    _current_template = None
    _theme_set = False
    _default_theme = "dark"

    def __new__(cls):
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
        cls.set_theme("default")  # Default theme

    @classmethod
    def list_themes(cls) -> list[str]:
        """List available themes."""
        return list(cls._available_themes.keys())

    @classmethod
    def set_theme(cls, theme_name: str):
        """Set the current theme."""

        # Use "default" theme
        if theme_name == "default":
            theme_name = cls._default_theme
        else:
            cls._theme_set = True

        # Check if the theme is in our available themes
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
        cls._log.info(f"Theme set to '{theme_name}'")

    @classmethod
    def data_bs_theme(cls) -> str:
        """Get the current Bootstrap `data-bs-theme` value."""
        return "dark" if "dark" in cls._current_theme_name.lower() else "light"

    @classmethod
    def current_theme(cls) -> str:
        """Get the name of the current theme."""
        return cls._current_theme_name

    @classmethod
    def colorscale(cls, scale_type: str = "sequential") -> list[list[float | str]]:
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
        theme = cls._available_themes[cls._current_theme_name]
        css_files = []

        # Add base.css or its CDN URL
        if theme["base_css"]:
            css_files.append(theme["base_css"])

        # Add the DBC template CSS
        css_files.append(cls._dbc_css)

        # Add custom.css if it exists
        if theme["custom_css"]:
            css_files.append("/custom.css")

        return css_files

    @classmethod
    def register_css_route(cls, app):
        """Register Flask route for custom.css."""

        @app.server.route("/custom.css")
        def serve_custom_css():
            theme = cls._available_themes[cls._current_theme_name]
            if theme["custom_css"]:
                return send_from_directory(theme["custom_css"].parent, theme["custom_css"].name)
            return "", 404

    @classmethod
    def _get_base_css_url(cls, theme_dir: Path) -> str:
        """
        Get the base CSS URL for a theme.

        Args:
            theme_dir (Path): The directory of the theme.

        Returns:
            str: The URL or file path for the base CSS.
        """
        base_css_file = theme_dir / "base_css.url"
        if base_css_file.exists():
            with open(base_css_file, "r") as f:
                content = f.read().strip()
                # Check if the content is a Dash Bootstrap theme name
                if content.startswith("dbc.themes."):
                    theme_name = content.split(".")[-1]
                    try:
                        # Dynamically get the URL from Dash Bootstrap Components
                        return getattr(dbc.themes, theme_name.upper())
                    except AttributeError:
                        cls._log.error(f"Invalid theme name in {base_css_file}: {theme_name}")
                # Otherwise, assume it's a direct URL
                elif content.startswith("http"):
                    return content
        return None

    @classmethod
    def _load_themes(cls):
        """Load available themes."""
        if not cls._themes_dir:
            cls._log.warning("No themes directory specified.")
            return

        for theme_dir in cls._themes_dir.iterdir():
            if theme_dir.is_dir():
                theme_name = theme_dir.name

                # Grab the base.css URL
                base_css_url = cls._get_base_css_url(theme_dir)

                # Find the first JSON file in the directory
                json_files = list(theme_dir.glob("*.json"))
                plotly_template = json_files[0] if json_files else None

                # Check for a custom.css file
                custom_css = theme_dir / "custom.css"

                cls._available_themes[theme_name] = {
                    "base_css": base_css_url,
                    "plotly_template": plotly_template,
                    "custom_css": custom_css if custom_css.exists() else None,
                }

        if not cls._available_themes:
            cls._log.warning(f"No themes found in '{cls._themes_dir}'.")


if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager()
    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())

    theme_manager.set_theme("light")
    print("Theme switched to:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())
