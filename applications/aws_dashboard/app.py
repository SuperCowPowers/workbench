"""SageWorks Dashboard: A SageWorks Web Application for viewing and managing SageWorks Artifacts"""

from dash import Dash, html, page_container, Input, Output
import dash_bootstrap_components as dbc


# SageWorks Imports
from sageworks.utils.plugin_manager import PluginManager
from sageworks.utils.theme_manager import ThemeManager


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Set up the Theme Manager
tm = ThemeManager()
tm.set_theme("dark")
css_files = tm.css_files()
print(css_files)

# Create the Dash app
app = Dash(
    __name__,
    title="SageWorks Dashboard",
    use_pages=True,
    external_stylesheets=css_files,
)

# Register the CSS route in the ThemeManager
tm.register_css_route(app)

# Note: The 'server' object is required for running the app with NGINX/uWSGI
server = app.server

# For Multi-Page Applications, we need to create a 'page container' to hold all the pages
# app.layout = html.Div([page_container])
app.layout = html.Div(
    [
        dbc.Container([page_container], fluid=True, className="dbc dbc-ag-grid"),
    ],
    **{"data-bs-theme": tm.data_bs_theme()},
)

# This bit of code is used for the subpage navigation


# Register the client-side callback
app.clientside_callback(
    """
    function(children) {
        if (children === "click") {
            const link = document.getElementById("navigate-link");
            if (link) link.click();  // Trigger the click
        }
        return null;  // No UI update
    }
    """,
    Output("trigger_link_click", "children"),  # Updated ID for clarity
    Input("navigate-link", "children"),
)

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
