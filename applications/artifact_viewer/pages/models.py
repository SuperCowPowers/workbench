import os
from dash import Dash, register_page
import dash
import dash_bootstrap_components as dbc


# SageWorks Imports
from sageworks.web_components import confusion_matrix, table, scatter_plot
from sageworks.web_components import (
    feature_importance,
    model_data,
    model_details,
    feature_details,
)

# Local Imports
from pages.layout.models_layout import models_layout
import pages.callbacks.models_callbacks as callbacks

register_page(__name__, path='/models')

# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)


# Read in our model data
data_path = os.path.join(os.path.dirname(__file__), "data/toy_data.csv")
model_info = model_data.ModelData(data_path)

# Create our components
model_df = model_info.get_model_df()
model_table = table.create("model_table", model_df, show_columns=["model_name", "date_created", "f_scores"])
details = model_details.create(model_info.get_model_details(0))
c_matrix = confusion_matrix.create(model_info.get_model_confusion_matrix(0))
scatter = scatter_plot.create(model_df)
my_feature_importance = feature_importance.create(model_info.get_model_feature_importance(0))
my_feature_details = feature_details.create(model_info.get_model_feature_importance(0))
components = {
    "model_table": model_table,
    "model_details": details,
    "confusion_matrix": c_matrix,
    "scatter_plot": scatter,
    "feature_importance": my_feature_importance,
    "feature_details": my_feature_details,
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.table_row_select(app, "model_table")
callbacks.update_figures(app, model_info)
callbacks.update_model_details(app, model_info)
callbacks.update_feature_details(app, model_info)


# Set up our layout (Dash looks for a var called layout)
layout = models_layout(components)
