from pathlib import Path
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template


# SageWorks Imports

from sageworks.web_components import confusion_matrix, table, scatter_plot
from sageworks.web_components import (
    mock_feature_importance,
    mock_model_data,
    mock_model_details,
    mock_feature_details,
)
from sageworks.views.model_web_view import ModelWebView

# Local Imports
from .layout import models_layout
from . import callbacks

register_page(
    __name__,
    path="/models",
    name="SageWorks - Models",
)

# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the FeatureSets in SageWorks
model_broker = ModelWebView()
models_rows = model_broker.models_summary()

# Read in our fake model data
data_path = str(Path(__file__).resolve().parent.parent / "data/toy_data.csv")
fake_model_info = mock_model_data.ModelData(data_path)

# Create a table to display the models
models_table = table.create(
    "models_table",
    models_rows,
    header_color="rgb(60, 100, 60)",
    row_select="single",
    markdown_columns=["Model Group"],
)

# Create all the other components on this page
model_df = fake_model_info.get_model_df()
details = mock_model_details.create(fake_model_info.get_model_details(0))
c_matrix = confusion_matrix.create(fake_model_info.get_model_confusion_matrix(0))
scatter = scatter_plot.create("model performance", model_df)
my_feature_importance = mock_feature_importance.create(fake_model_info.get_model_feature_importance(0))
my_feature_details = mock_feature_details.create(fake_model_info.get_model_feature_importance(0))
components = {
    "models_table": models_table,
    "model_details": details,
    "confusion_matrix": c_matrix,
    "scatter_plot": scatter,
    "feature_importance": my_feature_importance,
    "feature_details": my_feature_details,
}

# Set up our layout (Dash looks for a var called layout)
layout = models_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.refresh_data_timer(app)
callbacks.update_models_table(app, model_broker)

# Callback for the model table
callbacks.table_row_select(app, "models_table")
callbacks.update_figures(app, fake_model_info)
callbacks.update_model_details(app, fake_model_info)
callbacks.update_feature_details(app, fake_model_info)
