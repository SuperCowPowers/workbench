"""Compound Explorer Application"""

from dash import Dash, html
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.utils.theme_manager import ThemeManager
from workbench.web_interface.components.plugins import scatter_plot, molecule_viewer, generated_compounds

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

# Create the main components for the Compound Explorer
scatter_plot = scatter_plot.ScatterPlot(show_axes=False)
scatter_plot_component = scatter_plot.create_component("compound_scatter_plot")
molecule_view = molecule_viewer.MoleculeViewer()
molecule_view_component = molecule_view.create_component("molecule_view")
gen_compounds = generated_compounds.GeneratedCompounds()
gen_compounds_component = gen_compounds.create_component("gen_compounds")


# Create our components
components = {
    "scatter_plot": scatter_plot_component,
    "molecule_view": molecule_view_component,
    "gen_compounds": gen_compounds_component,
}

# Set up our application layout
app.layout = html.Div(
    [
        dbc.Container(compound_explorer_layout(**components), fluid=True, className="dbc dbc-ag-grid"),
    ],
    **{"data-bs-theme": tm.data_bs_theme()},
)

# Set up our application callbacks
callbacks.populate_scatter_plot(scatter_plot)
callbacks.hover_tooltip_callbacks()

# Set up the plugin callbacks
callbacks.setup_plugin_callbacks([molecule_view, gen_compounds])


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    # app.run(host="0.0.0.0", port=8000, debug=True)
    app.run(host="0.0.0.0", port=8000)
