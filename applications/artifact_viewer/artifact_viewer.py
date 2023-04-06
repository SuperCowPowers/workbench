"""Artifact Viewer: A SageWorks Application for viewing and managing SageWorks Artifacts"""
from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary

# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Create our Dash Application
# app = Dash(title='SageWorks: Artifacts', external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
# app = Dash(title="SageWorks: Artifacts", use_pages=True)
app = Dash(title='SageWorks: Artifacts', external_stylesheets=[dbc.themes.DARKLY], use_pages=True)
server = app.server

# This is the 'container' that will hold our page content (whichever page is selected/activated)
# app.layout = dash.page_container

"""
app.layout = html.Div(
    children=[
        dcc.Store(id='some-data'),
        dbc.Container(
            id='root-content',
            children=[
                # Your actual app and its pages are within a container of some sort.
                # In this case a Bootstrap container, but could also just be a div if
                # you are not using Bootstrap. The key point is
                # that the 'some-data' store is always in the layout,
                # irrespective of which page is active.
            ],
        ),
    ])
"""
my_data = {'foo': 'bar'}

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
# sageworks_artifacts = ArtifactsSummary()
# artifacts_summary = sageworks_artifacts.view_data()

app.layout = html.Div(
    children=[
        dcc.Store(id='sageworks_artifacts', data=my_data),
        dcc.Store(id='artifacts_summary', data=my_data),
        dash.page_container,
    ])


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    app.run_server(host="0.0.0.0", port=8080, debug=True)
