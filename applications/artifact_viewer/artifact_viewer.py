"""Artifact Viewer: A SageWorks Application for viewing and managing SageWorks Artifacts"""
from dash import Dash, html
import dash
from dash_bootstrap_components import themes


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Create our Dash Application
# app = Dash(title='SageWorks: Artifacts', external_stylesheets=[dbc.themes.BOOTSTRAP])
app = Dash(title="SageWorks: Artifacts", use_pages=True)
# app = Dash(title='SageWorks: Artifacts', external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# This is the 'container' that will hold our page content (whichever page is selected/activated)
app.layout = dash.page_container


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    app.run_server(host="0.0.0.0", port=8080, debug=True)
