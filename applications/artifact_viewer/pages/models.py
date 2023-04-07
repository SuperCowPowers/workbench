import os
from dash import register_page
import dash


# SageWorks Imports

from sageworks.web_components import confusion_matrix, table, scatter_plot
from sageworks.web_components import feature_importance, model_data, model_details, feature_details
from sageworks.views.artifacts_summary import ArtifactsSummary

# Local Imports
from pages.layout.models_layout import models_layout
import pages.callbacks.models_callbacks as callbacks

register_page(__name__, path='/models')

# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsSummary()
models_summary = sageworks_artifacts.models_summary(concise=True)

# Read in our model data
data_path = os.path.join(os.path.dirname(__file__), "data/toy_data.csv")
model_info = model_data.ModelData(data_path)

# Create a table to display the models
models_table = table.create("models_table", models_summary, header_color="rgb(60, 100, 60)",
                            markdown_columns=["Model Group"], row_select="single")

# Create our components
model_df = model_info.get_model_df()
# models_table = table.create("models_table", model_df, show_columns=["model_name", "date_created", "f_scores"])
details = model_details.create(model_info.get_model_details(0))
c_matrix = confusion_matrix.create(model_info.get_model_confusion_matrix(0))
scatter = scatter_plot.create(model_df)
my_feature_importance = feature_importance.create(model_info.get_model_feature_importance(0))
my_feature_details = feature_details.create(model_info.get_model_feature_importance(0))
components = {
    "models_table": models_table,
    "model_details": details,
    "confusion_matrix": c_matrix,
    "scatter_plot": scatter,
    "feature_importance": my_feature_importance,
    "feature_details": my_feature_details,
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_models_table(app, sageworks_artifacts)

# Callback for the model table
# callbacks.table_row_select(app, "models_table")
callbacks.update_figures(app, model_info)
callbacks.update_model_details(app, model_info)
callbacks.update_feature_details(app, model_info)

# Set up our layout (Dash looks for a var called layout)
layout = models_layout(components)
