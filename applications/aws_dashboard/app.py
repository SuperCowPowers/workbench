"""Workbench Dashboard: A Workbench Web Application for viewing and managing Workbench Artifacts"""

from dash import Dash, html, dcc, page_container
import dash_bootstrap_components as dbc
from dash import page_registry

# Workbench Imports
from workbench.utils.plugin_manager import PluginManager
from workbench.utils.theme_manager import ThemeManager

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
)

# Register the CSS route in the ThemeManager
tm.register_css_route(app)

# Note: The 'server' object is required for running the app with NGINX/uWSGI
server = app.server

# For Multi-Page Applications, we need to create a 'page container' to hold all the pages
plugin_info_id = "plugin-pages-info"
app.layout = html.Div(
    [
        # URL for subpage navigation (jumping to feature_sets, models, etc.)
        dcc.Location(id="url", refresh="callback-nav"),
        dcc.Store(id=plugin_info_id, data={}),
        dbc.Container([page_container], fluid=True, className="dbc dbc-ag-grid"),
    ],
    **{"data-bs-theme": tm.data_bs_theme()},
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
