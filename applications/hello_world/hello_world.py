"""HelloWorld: A SageWorks HelloWorld Application"""
from dash import Dash
import dash_bootstrap_components as dbc


# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary
from sageworks.web_components import confusion_matrix, table, scatter_plot
from sageworks.web_components import feature_importance, model_data, model_details, feature_details

# Local Imports
import layout
import callbacks


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Create our Dash Application
app = Dash(title='SageWorks Artifacts', external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = Dash(title='Hello World Application', external_stylesheets=[dbc.themes.DARKLY])
server = app.server


def setup_artifact_viewer():

    # Set Default Template for figures
    # load_figure_template('darkly')

    # Grab a view that gives us a summary of all the artifacts currently in SageWorks
    sageworks_artifacts = ArtifactsSummary()
    artifacts_summary = sageworks_artifacts.view_data()

    # Just a bunch of tables for now :)
    tables = {}
    for service_category, artifact_info_df in artifacts_summary.items():

        # Grab the Artifact Information DataFrame for each AWS Service
        tables[service_category] = table.create(service_category, artifact_info_df)

    # Create our components
    components = {
        'incoming_data': tables['INCOMING_DATA'],
        'data_sources': tables['DATA_SOURCES'],
        'feature_sets': tables['FEATURE_SETS'],
        'models': tables['MODELS'],
        'endpoints': tables['ENDPOINTS']
    }

    # Setup up our application layout
    app.layout = layout.artifact_layout(app, components)

    # Setup our callbacks/connections
    """
    callbacks.table_row_select(app, 'model_table')
    callbacks.update_figures(app, df)
    callbacks.update_model_details(app, df)
    callbacks.update_feature_details(app, df)
    """


# Now actually set up the scoreboard
setup_artifact_viewer()


if __name__ == '__main__':
    # Run our web application in TEST mode
    # Note: This 'main' is purely for running/testing locally
    app.run_server(host='0.0.0.0', port=8080, debug=True)
