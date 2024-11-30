"""SageWorks Dashboard: A SageWorks Web Application for viewing and managing SageWorks Artifacts"""

import os
import json
import plotly.io as pio
from dash import Dash, page_container, html
import dash_bootstrap_components as dbc
from sageworks.utils.plugin_manager import PluginManager


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Load our custom template (themes for Plotly figures)
with open("assets/dark_custom.json", "r") as f:
    custom_template = json.load(f)

# Register the custom template (themes for Plotly figures)
pio.templates["dark_custom"] = custom_template

# Set as the default template
pio.templates.default = "dark_custom"

# Spin up our Plugin Manager
pm = PluginManager()

# Custom CSS
custom_css_files = pm.get_css_files()

# Load our custom CSS files into the Assets folder
"""
for css_file in custom_css_files:
    shutil.copy(css_file, "assets/")
"""

# Get basename of the CSS files
css_files = [os.path.basename(css_file) for css_file in custom_css_files]

# Create our Dash Application
app = Dash(
    __name__,
    title="SageWorks Dashboard",
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY]
)
server = app.server

# For Multi-Page Applications, we need to create a 'page container' to hold all the pages
app.layout = html.Div([page_container])

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
