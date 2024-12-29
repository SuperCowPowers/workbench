import json
import logging
from importlib.resources import files
from pathlib import Path
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.api import ParameterStore
from workbench.utils.color_utils import color_to_rgba


class ThemeManager:
    _instance = None  # Singleton instance

    # Class-level state
    _log = logging.getLogger("workbench")
    _themes_path = None
    _dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    _available_themes = {}
    _current_theme_name = None
    _current_template = None
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
        config_theme_path = cm.get_config("WORKBENCH_THEMES")

        # If the theme path is not set, use the default path
        if not config_theme_path:
            cls._log.important("Using default themes path...")
            cls._theme_path = files("workbench") / "themes"
        else:
            cls._theme_path = Path(config_theme_path)

        # Make sure the themes path exists
        if not cls._theme_path.exists():
            cls._log.error(f"The themes path '{cls._theme_path}' does not exist.")

        # Our parameter store
        cls._ps = ParameterStore()

        # Load the available themes and set the automatic theme
        cls._load_themes()
        cls.set_theme("auto")

    @classmethod
    def list_themes(cls) -> list[str]:
        """List available themes."""
        return list(cls._available_themes.keys())

    @classmethod
    def set_theme(cls, theme_name: str):
        """Set the current theme."""

        # For 'auto', we try to grab a theme from the Parameter Store
        # if we can't find one, we'll set the theme to the default
        if theme_name == "auto":
            theme_name = cls._ps.get("/workbench/dashboard/theme", warn=False) or cls._default_theme

        # Check if the theme is in our available themes
        if theme_name not in cls._available_themes:
            cls._log.error(f"Theme '{theme_name}' is not available, using default theme.")
            theme_name = cls._default_theme

        # Grab the theme from the available themes
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
    def dark_mode(cls) -> bool:
        """Check if the current theme is a dark mode theme."""
        return "dark" in cls.current_theme().lower()

    @classmethod
    def data_bs_theme(cls) -> str:
        """Get the current Bootstrap `data-bs-theme` value."""
        return "dark" if cls.dark_mode() else "light"

    @classmethod
    def current_theme(cls) -> str:
        """Get the name of the current theme."""
        return cls._current_theme_name

    @classmethod
    def background(cls) -> list[list[float | str]]:
        """Get the plot background for the current theme."""

        # We have 2 background options (paper_bgcolor and plot_bgcolor)
        background = cls._current_template["layout"]["paper_bgcolor"]
        background = cls._current_template["layout"]["plot_bgcolor"]
        return color_to_rgba(background)

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

    @staticmethod
    def adjust_colorscale_alpha(colorscale, alpha=0.5):
        """
        Adjust the alpha value of the first color in the colorscale.

        Args:
            colorscale (list): The colorscale list with format [[value, color], ...].
            alpha (float): The new alpha value for the first color (0 to 1).

        Returns:
            list: The updated colorscale.
        """
        updated_colorscale = colorscale.copy()

        if updated_colorscale and "rgba" in updated_colorscale[0][1]:
            # Parse the existing RGBA value and modify alpha
            rgba_values = updated_colorscale[0][1].strip("rgba()").split(",")
            rgba_values[-1] = str(alpha)  # Update the alpha channel
            updated_colorscale[0][1] = f"rgba({','.join(rgba_values)})"

        return updated_colorscale

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
        if not cls._theme_path:
            cls._log.warning("No themes path specified...")
            return

        for theme_dir in cls._theme_path.iterdir():
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
            cls._log.warning(f"No themes found in '{cls._theme_path}'...")


if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager()
    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())

    theme_manager.set_theme("light")
    print("Theme switched to:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())
