"""SageWorks Dashboard: A SageWorks Web Application for viewing and managing SageWorks Artifacts"""

from dash import Dash, page_container
import dash_bootstrap_components as dbc
import shutil

# SageWorks Imports
from sageworks.utils.plugin_manager import PluginManager
from sageworks.utils.theme_manager import ThemeManager


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Set up the Theme Manager
tm = ThemeManager(default_theme="dark")
css_files = tm.get_current_css_files()

# Copy custom CSS to the assets directory
assets_dir = Path(__file__).parent / "assets"
assets_dir.mkdir(exist_ok=True)

for css_file in tm.get_current_css_files():
    if not css_file.startswith("http"):  # Skip external URLs
        shutil.copy(css_file, assets_dir / Path(css_file).name)

# Create our Dash Application
app = Dash(
    __name__,
    title="SageWorks Dashboard",
    use_pages=True,
    external_stylesheets=css_files,
)
server = app.server

# For Multi-Page Applications, we need to create a 'page container' to hold all the pages
# app.layout = html.Div([page_container])
app.layout = dbc.Container([page_container], fluid=True, className="dbc")

# Spin up the Plugin Manager
pm = PluginManager()

# Grab any plugin pages
plugin_pages = pm.get_pages()

# Setup each if the plugin pages (call layout and callbacks internally)
for name, page in plugin_pages.items():
    page.page_setup(app)


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    # app.run(host="0.0.0.0", port=8000, debug=True)
    app.run(host="0.0.0.0", port=8000)
