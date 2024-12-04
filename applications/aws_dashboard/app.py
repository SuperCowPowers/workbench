"""SageWorks Dashboard: A SageWorks Web Application for viewing and managing SageWorks Artifacts"""

import os
import json
import plotly.io as pio
from dash import Dash, page_container
import dash_bootstrap_components as dbc
from sageworks.utils.plugin_manager import PluginManager


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code


# Hardcoded theme selection
USE_DARK_THEME = True

# Determine the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set Plotly template
template_file = (
    os.path.join(current_dir, "assets", "darkly_custom.json")
    if USE_DARK_THEME
    else os.path.join(current_dir, "assets", "flatly.json")
)
with open(template_file, "r") as f:
    template = json.load(f)

pio.templates["custom_template"] = template
pio.templates.default = "custom_template"

"""
# Dynamically set the Bootstrap theme
bootstrap_theme = dbc.themes.DARKLY if USE_DARK_THEME else dbc.themes.FLATLY

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
"""

# Spin up the Plugin Manager
pm = PluginManager()

# Load any custom CSS files
custom_css_files = pm.get_css_files()
css_files = [dbc.themes.DARKLY]  # , dbc_css]
css_files.extend(custom_css_files)

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
