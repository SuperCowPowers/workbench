import os

import pandas as pd
from dash import Dash
import dash_bootstrap_components as dbc


# SageWorks Imports
from sageworks.views.artifacts import Artifacts
from sageworks.web_interfaces import layout, callbacks
from sageworks.web_interfaces.components import confusion_matrix, table, scatter_plot
from sageworks.web_interfaces.components import feature_importance, model_data, model_details, feature_details


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Create our Dash Application
app = Dash(title='SageWork Artifacts', external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = Dash(title='Hello World Application', external_stylesheets=[dbc.themes.DARKLY])
server = app.server


def setup_artifact_viewer():

    # Set Default Template for figures
    # load_figure_template('darkly')

    # Grab a view that gives us a summary of all the artifacts currently in SageWorks
    sageworks_artifacts = Artifacts()
    sageworks_artifacts.refresh()
    artifact_summary = sageworks_artifacts.view_data()

    """
    {<ServiceCategory.INCOMING_DATA: 1>: ['aqsol_public_data.csv'],
 <ServiceCategory.DATA_CATALOG: 2>: ['athena_input_data',
                                     'sagemaker_featurestore',
                                     'sageworks',
                                     'temp_crawler_output'],
 <ServiceCategory.FEATURE_STORE: 3>: ['test-feature-set', 'AqSolDB-base'],
 <ServiceCategory.MODELS: 4>: ['test-model', 'Solubility-Models'],
 <ServiceCategory.ENDPOINTS: 5>: ['jumpstart-dft-bert-movie-reviews',
                                  'solubility-base-endpoint']}
    """

    # Just a bunch of tables for now :)
    for service_category, artifact_list in artifact_summary.items():

        # Make a baby dataframe for the table
        df = pd.DataFrame({"Artifact Name": artifact_list})
        _table = table.create(service_category, df)

    # Create our components
    """
    model_df = model_info.get_model_df()
    model_table = table.create('model_table', model_df, show_columns=['model_name', 'date_created', 'f_scores'])
    details = model_details.create(model_info.get_model_details(0))
    c_matrix = confusion_matrix.create(model_info.get_model_confusion_matrix(0))
    scatter = scatter_plot.create(model_df)
    my_feature_importance = feature_importance.create(model_info.get_model_feature_importance(0))
    my_feature_details = feature_details.create(model_info.get_model_feature_importance(0))
    """
    components = {
        'model_table': model_table,
        'model_details': details,
        'confusion_matrix': c_matrix,
        'scatter_plot': scatter,
        'feature_importance': my_feature_importance,
        'feature_details': my_feature_details
    }

    # Setup up our application layout
    app.layout = layout.scoreboard_layout(app, components)

    # Setup our callbacks/connections
    callbacks.table_row_select(app, 'model_table')
    callbacks.update_figures(app, model_info)
    callbacks.update_model_details(app, model_info)
    callbacks.update_feature_details(app, model_info)


# Now actually set up the scoreboard
setup_artifact_viewer()


if __name__ == '__main__':
    # Run our web application in TEST mode
    # Note: This 'main' is purely for running/testing locally
    app.run_server(host='0.0.0.0', port=8080, debug=True)
