"""Workbench Dashboard: A Workbench Web Application for viewing and managing Workbench Artifacts"""

from dash import Dash, html, dcc, page_container, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash import page_registry

# Workbench Imports
from workbench.utils.plugin_manager import PluginManager
from workbench.utils.theme_manager import ThemeManager
from workbench.web_interface.components.settings_menu import SettingsMenu

# Set up the logging
import logging

log = logging.getLogger("workbench")


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Set up the Theme Manager
tm = ThemeManager()
tm.set_theme("auto")
css_files = tm.css_files()
print(css_files)

# Set the Dash App Title
app_title = tm.branding().get("app_title", "Workbench Dashboard")

# Create the Dash app
app = Dash(
    __name__,
    title=app_title,
    use_pages=True,
    external_stylesheets=css_files,
    assets_folder="assets",
)

# Register the CSS route in the ThemeManager
tm.register_css_route(app)

# Custom index string to sync localStorage theme to cookie on page load
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
            // Sync localStorage theme to cookie on page load (before Flask renders)
            (function() {
                var theme = localStorage.getItem('wb_theme');
                if (theme) {
                    document.cookie = 'wb_theme=' + theme + '; path=/; max-age=31536000';
                }
            })();
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Note: The 'server' object is required for running the app with WSGI servers
server = app.server

# ASGI wrapper for Uvicorn (only needed in Docker/production)
try:
    from asgiref.wsgi import WsgiToAsgi

    asgi_app = WsgiToAsgi(server)
except ImportError:
    asgi_app = None  # Not available when running locally with `python app.py`

# Create the settings menu component
settings_menu = SettingsMenu()
settings_menu_id = "settings-menu"

# For Multi-Page Applications, we need to create a 'page container' to hold all the pages
plugin_info_id = "plugin-pages-info"
app.layout = html.Div(
    [
        # URL for subpage navigation (jumping to feature_sets, models, etc.)
        dcc.Location(id="url", refresh="callback-nav"),
        dcc.Store(id=plugin_info_id, data={}),
        # Theme store for dynamic theme switching (plugins can use this as Input to re-render figures)
        dcc.Store(id="workbench-theme-store", data=tm.current_theme()),
        # Settings menu in top-right corner
        html.Div(
            settings_menu.create_component(settings_menu_id),
            style={"position": "absolute", "top": "25px", "right": "20px", "zIndex": 1000},
        ),
        dbc.Container([page_container], fluid=True, className="dbc dbc-ag-grid"),
    ],
    **{"data-bs-theme": tm.data_bs_theme()},
)

# Clientside callback for theme switching (stores in localStorage, sets cookie, triggers checkmark update)
app.clientside_callback(
    settings_menu.get_clientside_callback_code(),
    Output(f"{settings_menu_id}-init", "data"),
    Input({"type": f"{settings_menu_id}-theme-item", "theme": ALL}, "n_clicks"),
    State({"type": f"{settings_menu_id}-theme-item", "theme": ALL}, "id"),
)

# Clientside callback to update checkmarks based on localStorage
app.clientside_callback(
    settings_menu.get_checkmark_callback_code(),
    Output({"type": f"{settings_menu_id}-checkmark", "theme": ALL}, "children"),
    Input(f"{settings_menu_id}-init", "data"),
    State({"type": f"{settings_menu_id}-checkmark", "theme": ALL}, "id"),
)

# Clientside callback to update the workbench-theme-store (plugins listen to this for figure re-renders)
app.clientside_callback(
    "function(theme) { return theme; }",
    Output("workbench-theme-store", "data"),
    Input(f"{settings_menu_id}-init", "data"),
)

# Spin up the Plugin Manager
pm = PluginManager()

# Grab any plugin pages
plugin_pages = pm.get_pages()

# Setup each if the plugin pages (call layout and callbacks internally)
for name, page in plugin_pages.items():
    # Note: We need to catch any exceptions here
    try:
        page.page_setup(app)
    except Exception as e:
        # Put the exception full stack trace
        log.critical(f"Error setting up plugin page: {name}")
        log.critical(e, exc_info=True)
        continue
log.info("Done with Plugin Pages")

# Grab our plugin page info from the page registry and populate our plugin-pages-info store
dashboard_page_paths = [
    "/",
    "/data_sources",
    "/feature_sets",
    "/models",
    "/endpoints",
    "/pipelines",
    "/license",
    "/status",
]

# Pull the plugin pages path and name
plugin_pages = {}
for page_id, page_info in page_registry.items():
    if page_info["path"] not in dashboard_page_paths:
        plugin_pages[page_info["path"]] = page_info["name"]
print(plugin_pages)

# Update the plugin-pages-info store
for component in app.layout.children:
    if isinstance(component, dcc.Store) and component.id == plugin_info_id:
        component.data = plugin_pages


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    # app.run(host="0.0.0.0", port=8000, debug=True)
    app.run(host="0.0.0.0", port=8000)
