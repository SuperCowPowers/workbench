import os
import shutil
import json
import logging
from importlib.resources import files
from pathlib import Path
import atexit
import tempfile
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory
from typing import Optional, List, Union

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.api import ParameterStore
from workbench.utils.color_utils import color_to_rgba
from workbench.utils.s3_utils import copy_s3_files_to_local


class ThemeManager:
    _instance = None  # Singleton instance

    # Class-level state
    log = logging.getLogger("workbench")
    theme_path_list = [files("workbench") / "themes"]
    dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    available_themes = {}
    current_theme_name = None
    current_template = None
    default_theme = "dark"
    ps = ParameterStore()
    loading_temp_dir = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize()  # Initialize class-level state
        return cls._instance

    @classmethod
    def _initialize(cls):
        """Initialize the ThemeManager's shared state."""
        cm = ConfigManager()
        custom_themes = cm.get_config("WORKBENCH_THEMES")

        # Check if they set custom themes dir/S3 path
        if custom_themes:

            # Check if this is an S3 path
            if custom_themes.startswith("s3://"):
                cls.loading_temp_dir = tempfile.mkdtemp()
                cls.log.important(f"Moving {custom_themes} themes to local path {cls.loading_temp_dir}")
                copy_s3_files_to_local(custom_themes, cls.loading_temp_dir)
                atexit.register(cls._cleanup_temp_dir)
                custom_themes = cls.loading_temp_dir

            # Make sure the themes path exists
            custom_path = Path(custom_themes)
            if not custom_path.exists():
                cls.log.error(f"The custom themes path '{custom_path}' does not exist.")
            else:
                # Add the custom path to the theme path
                cls.theme_path_list += [custom_path]

        # Load the available themes and set the automatic theme
        cls._load_themes()
        cls.set_theme("auto")

    @classmethod
    def list_themes(cls) -> list[str]:
        """List available themes."""
        return list(cls.available_themes.keys())

    @classmethod
    def set_theme(cls, theme_name: str):
        """Set the current theme."""

        # For 'auto', we check multiple sources in priority order:
        # 1. Browser cookie (from localStorage, for per-user preference)
        # 2. Parameter Store (for org-wide default)
        # 3. Default theme
        if theme_name == "auto":
            theme_name = None

            # 1. Check Flask request cookie (set from localStorage)
            try:
                from flask import request, has_request_context

                if has_request_context():
                    theme_name = request.cookies.get("wb_theme")
            except Exception:
                pass

            # 2. Fall back to ParameterStore
            if not theme_name:
                theme_name = cls.ps.get("/workbench/dashboard/theme", warn=False)

            # 3. Fall back to default
            theme_name = theme_name or cls.default_theme

        # Check if the theme is in our available themes
        if theme_name not in cls.available_themes:
            cls.log.error(f"Theme '{theme_name}' is not available, trying another theme...")
            theme_name = (
                cls.default_theme if cls.default_theme in cls.available_themes else list(cls.available_themes.keys())[0]
            )

        # Grab the theme from the available themes
        theme = cls.available_themes[theme_name]

        # Update Plotly template
        if theme["plotly_template"]:
            with open(theme["plotly_template"], "r") as f:
                cls.current_template = json.load(f)
            pio.templates["custom_template"] = cls.current_template
            pio.templates.default = "custom_template"
        else:
            cls.log.error(f"No Plotly template found for '{theme_name}'.")

        # Update the current theme name
        cls.current_theme_name = theme_name
        cls.log.info(f"Theme set to '{theme_name}'")

    # Bootstrap themes that are dark mode (from Bootswatch)
    _dark_bootstrap_themes = {"DARKLY", "CYBORG", "SLATE", "SOLAR", "SUPERHERO", "VAPOR"}

    @classmethod
    def dark_mode(cls) -> bool:
        """Check if the current theme is a dark mode theme.

        Determines dark mode by checking if the Bootstrap base theme is a known dark theme.
        Falls back to checking if 'dark' is in the theme name.
        """
        theme = cls.available_themes.get(cls.current_theme_name, {})
        base_css = theme.get("base_css", "")

        # Check if the base CSS URL contains a known dark Bootstrap theme
        if base_css:
            base_css_upper = base_css.upper()
            for dark_theme in cls._dark_bootstrap_themes:
                if dark_theme in base_css_upper:
                    return True

        # Fallback: check if 'dark' is in the theme name
        return "dark" in cls.current_theme().lower()

    @classmethod
    def data_bs_theme(cls) -> str:
        """Get the current Bootstrap `data-bs-theme` value."""
        return "dark" if cls.dark_mode() else "light"

    @classmethod
    def current_theme(cls) -> str:
        """Get the name of the current theme."""
        return cls.current_theme_name

    @classmethod
    def get_theme_urls(cls) -> dict:
        """Get a mapping of theme names to their Bootstrap CSS URLs."""
        return {name: theme["base_css"] for name, theme in cls.available_themes.items() if theme["base_css"]}

    @classmethod
    def get_dark_themes(cls) -> list:
        """Get list of theme names that are dark mode themes."""
        dark_themes = []
        for name, theme in cls.available_themes.items():
            base_css = theme.get("base_css", "")
            if base_css:
                base_css_upper = base_css.upper()
                for dark_theme in cls._dark_bootstrap_themes:
                    if dark_theme in base_css_upper:
                        dark_themes.append(name)
                        break
                else:
                    # Fallback: check if 'dark' is in the theme name
                    if "dark" in name.lower():
                        dark_themes.append(name)
        return dark_themes

    @classmethod
    def get_theme_switch_js(cls) -> str:
        """Get the JavaScript code for dynamic theme switching.

        This code should be used in a clientside callback that updates the Bootstrap
        stylesheet and data-bs-theme attribute when the user selects a new theme.
        """
        theme_urls = cls.get_theme_urls()
        dark_themes = cls.get_dark_themes()

        return f"""
        function(n_clicks_list, ids) {{
            // Use callback context to find which input triggered this callback
            const ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {{
                return window.dash_clientside.no_update;
            }}

            // Get the triggered input's id (it's a JSON string for pattern-matching callbacks)
            const triggeredId = ctx.triggered[0].prop_id.split('.')[0];
            let clickedTheme = null;

            try {{
                const parsedId = JSON.parse(triggeredId);
                clickedTheme = parsedId.theme;
            }} catch (e) {{
                // Not a pattern-matching callback ID
                return window.dash_clientside.no_update;
            }}

            if (!clickedTheme) {{
                return window.dash_clientside.no_update;
            }}

            // Store in localStorage and cookie
            localStorage.setItem('wb_theme', clickedTheme);
            document.cookie = `wb_theme=${{clickedTheme}}; path=/; max-age=31536000`;

            // Theme URL mapping (generated from ThemeManager)
            const themeUrls = {json.dumps(theme_urls)};
            const darkThemes = {json.dumps(dark_themes)};

            // Update Bootstrap's data-bs-theme attribute
            const bsTheme = darkThemes.includes(clickedTheme) ? 'dark' : 'light';
            document.documentElement.setAttribute('data-bs-theme', bsTheme);

            // Find and update the Bootstrap stylesheet
            const newUrl = themeUrls[clickedTheme];
            if (newUrl) {{
                let stylesheet = document.querySelector('link[href*="bootswatch"]') ||
                                 document.querySelector('link[href*="bootstrap"]') ||
                                 document.querySelector('link[href="/base.css"]');
                if (stylesheet) {{
                    stylesheet.setAttribute('href', newUrl);
                }} else {{
                    stylesheet = document.createElement('link');
                    stylesheet.rel = 'stylesheet';
                    stylesheet.href = newUrl;
                    document.head.appendChild(stylesheet);
                }}
            }}

            // Force reload custom.css with cache-busting query param
            let customCss = document.querySelector('link[href^="/custom.css"]');
            if (customCss) {{
                customCss.setAttribute('href', `/custom.css?t=${{Date.now()}}`);
            }}

            return clickedTheme;
        }}
        """

    @classmethod
    def background(cls) -> list[list[float | str]]:
        """Get the plot background for the current theme."""

        # We have 2 background options (paper_bgcolor and plot_bgcolor)
        background = cls.current_template["layout"]["paper_bgcolor"]
        background = cls.current_template["layout"]["plot_bgcolor"]
        return color_to_rgba(background)

    @classmethod
    def branding(cls) -> dict:
        """Get the branding for the current theme."""
        theme = cls.available_themes[cls.current_theme_name]
        branding = {}
        if theme["branding"]:
            with open(theme["branding"], "r") as f:
                branding = json.load(f)
        return branding

    @classmethod
    def colorscale(cls, scale_type: str = "sequential") -> Optional[List[List[Union[float, str]]]]:
        """Get the colorscale for the current theme."""

        # We have 3 colorscale here (diverging, sequential, and sequentialminus)
        color_scales = cls.current_template["layout"]["colorscale"]

        # We have another colorscale in data/heatmap
        color_scales["heatmap"] = cls.current_template["data"]["heatmap"][0]["colorscale"]
        if scale_type in color_scales:
            return color_scales[scale_type]
        else:
            # Use the default colorscale (sequential)
            try:
                cls.log.warning(
                    f"Color scale '{scale_type}' not found for template '{cls.current_theme_name}', returning default."
                )
                return color_scales["sequential"]
            except KeyError:
                cls.log.error(f"No color scales found for template '{cls.current_theme_name}'.")
        return None

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
        """Get the list of CSS files for the current theme.

        Note: Uses /base.css route for dynamic theme switching instead of CDN URLs.
        """
        css_files = []

        # Use Flask route for base CSS (allows dynamic theme switching)
        css_files.append("/base.css")

        # Add the DBC template CSS
        css_files.append(cls.dbc_css)

        # Add custom.css if it exists
        css_files.append("/custom.css")

        return css_files

    @classmethod
    def _get_theme_from_cookie(cls):
        """Get the theme dict based on the wb_theme cookie, falling back to current theme."""
        from flask import request

        theme_name = request.cookies.get("wb_theme")
        if theme_name and theme_name in cls.available_themes:
            return cls.available_themes[theme_name], theme_name
        return cls.available_themes[cls.current_theme_name], cls.current_theme_name

    @classmethod
    def register_css_route(cls, app):
        """Register Flask routes for CSS and before_request hook for theme switching."""
        from flask import redirect

        @app.server.before_request
        def check_theme_cookie():
            """Check for theme cookie on each request and update theme if needed."""
            _, theme_name = cls._get_theme_from_cookie()
            if theme_name != cls.current_theme_name:
                cls.set_theme(theme_name)

        @app.server.route("/base.css")
        def serve_base_css():
            """Redirect to the appropriate Bootstrap theme CSS based on cookie."""
            theme, _ = cls._get_theme_from_cookie()
            if theme["base_css"]:
                return redirect(theme["base_css"])
            return "", 404

        @app.server.route("/custom.css")
        def serve_custom_css():
            """Serve the custom.css file based on cookie."""
            theme, _ = cls._get_theme_from_cookie()
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
                        cls.log.error(f"Invalid theme name in {base_css_file}: {theme_name}")
                # Otherwise, assume it's a direct URL
                elif content.startswith("http"):
                    return content
        return None

    @classmethod
    def _load_themes(cls):
        """Load available themes."""
        if not cls.theme_path_list:
            cls.log.warning("No themes paths specified...")
            return

        # Loop over each path in the theme path
        for theme_path in cls.theme_path_list:
            for theme_dir in theme_path.iterdir():
                # Skip hidden directories (e.g., .idea, .git)
                if not theme_dir.is_dir() or theme_dir.name.startswith("."):
                    continue
                theme_name = theme_dir.name

                # Grab the base.css URL
                base_css_url = cls._get_base_css_url(theme_dir)

                # Grab the plotly template json, custom.css, and branding json
                plotly_template = theme_dir / "plotly.json"
                custom_css = theme_dir / "custom.css"
                branding = theme_dir / "branding.json"

                cls.available_themes[theme_name] = {
                    "base_css": base_css_url,
                    "plotly_template": plotly_template,
                    "custom_css": custom_css if custom_css.exists() else None,
                    "branding": branding if branding.exists() else None,
                }

        if not cls.available_themes:
            cls.log.warning(f"No themes found in '{cls.theme_path_list}'...")

    @classmethod
    def _cleanup_temp_dir(cls):
        """Cleans up the temporary directory created for S3 files."""
        if cls.loading_temp_dir and os.path.isdir(cls.loading_temp_dir):
            cls.log.important(f"Cleaning up temporary directory: {cls.loading_temp_dir}")
            shutil.rmtree(cls.loading_temp_dir)
            cls.loading_temp_dir = None


if __name__ == "__main__":
    # Example usage of the ThemeManager
    theme_manager = ThemeManager()
    print("Available Themes:", theme_manager.list_themes())
    print("Current Theme:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())

    theme_manager.set_theme("light")
    print("Theme switched to:", theme_manager.current_theme())
    print("CSS Files for Current Theme:", theme_manager.css_files())

    # Get the branding for the current theme
    print("Branding:", theme_manager.branding())

    # Get some color scales
    print("Colorscale:", theme_manager.colorscale())
    print("Sequential Colorscale:", theme_manager.colorscale("sequential"))
    print("Diverging Colorscale:", theme_manager.colorscale("diverging"))
    print("Heatmap Colorscale:", theme_manager.colorscale("heatmap"))
