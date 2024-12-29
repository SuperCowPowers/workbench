"""Compound Explorer Application"""

from dash import Dash

# Workbench Imports
from workbench.utils.theme_manager import ThemeManager
from workbench.web_interface.components.plugins import scatter_plot

# Local Imports
from layout import compound_explorer_layout
import callbacks


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Set up the Theme Manager
tm = ThemeManager()
tm.set_theme("auto")
css_files = tm.css_files()
print(css_files)

# Create the Dash app
app = Dash(
    __name__,
    title="Compound Explorer",
    external_stylesheets=css_files,
)

# Register the CSS route in the ThemeManager
tm.register_css_route(app)

# Note: The 'server' object is required for running the app with NGINX/uWSGI
server = app.server

# Create the main Compound plot
compound_plot = scatter_plot.ScatterPlot()
compound_plot_component = compound_plot.create_component("compound_scatter_plot")

# Create our components
components = {
    "compound_scatter_plot": compound_plot_component,
}

# Set up our application layout
app.layout = compound_explorer_layout(**components)

# Set up our application callbacks
callbacks.scatter_plot_callbacks(compound_plot)
callbacks.update_compound_diagram()


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    app.run(host="0.0.0.0", port=8000, debug=True)
    # app.run(host="0.0.0.0", port=8082)
