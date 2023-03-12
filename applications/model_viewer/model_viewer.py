import os
from dash import Dash
import dash_bootstrap_components as dbc


# SageWorks Imports
from sageworks.web_components import confusion_matrix, table, scatter_plot
from sageworks.web_components import feature_importance, model_data, model_details, feature_details

# Local Imports
import layout
import callbacks


# Note: The 'app' and 'server' objects need to be at the top level since NGINX/uWSGI needs to
#       import this file and use the server object as an ^entry-point^ into the Dash Application Code

# Create our Dash Application
app = Dash(title='Model Details Viewer', external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = Dash(title='Hello World Application', external_stylesheets=[dbc.themes.DARKLY])
server = app.server


def setup_model_details_view():

    # Set Default Template for figures
    # load_figure_template('darkly')

    # Read in our model data
    data_path = os.path.join(os.path.dirname(__file__), 'data/toy_data.csv')
    model_info = model_data.ModelData(data_path)

    # Create our components
    model_df = model_info.get_model_df()
    model_table = table.create('model_table', model_df, show_columns=['model_name', 'date_created', 'f_scores'])
    details = model_details.create(model_info.get_model_details(0))
    c_matrix = confusion_matrix.create(model_info.get_model_confusion_matrix(0))
    scatter = scatter_plot.create(model_df)
    my_feature_importance = feature_importance.create(model_info.get_model_feature_importance(0))
    my_feature_details = feature_details.create(model_info.get_model_feature_importance(0))
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


# Now actually set up the model details view
setup_model_details_view()


if __name__ == '__main__':
    # Run our web application in TEST mode
    # Note: This 'main' is purely for running/testing locally
    app.run_server(host='0.0.0.0', port=8080, debug=True)
