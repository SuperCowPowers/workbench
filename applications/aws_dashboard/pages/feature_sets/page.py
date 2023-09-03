"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import table, data_details_markdown, distribution_plots, heatmap, scatter_plot
from sageworks.views.feature_set_web_view import FeatureSetWebView
from sageworks.utils.pandas_utils import corr_df_from_artifact_info

# Local Imports
from .layout import feature_sets_layout
from . import callbacks

register_page(__name__, path="/feature_sets", name="SageWorks - Feature Sets")

# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the FeatureSets in SageWorks
feature_set_broker = FeatureSetWebView()
feature_set_rows = feature_set_broker.feature_sets_summary()

# Create a table to display the data sources
feature_sets_table = table.create(
    "feature_sets_table",
    feature_set_rows,
    header_color="rgb(100, 100, 60)",
    row_select="single",
    markdown_columns=["Feature Group"],
)

# Grab smart_sample rows from the first Feature Set
smart_sample_rows = feature_set_broker.feature_set_smart_sample(0)
feature_set_sample_rows = table.create(
    "feature_set_sample_rows",
    smart_sample_rows,
    header_color="rgb(60, 60, 100)",
    max_height="300px",
    color_column="outlier_group",
)

# Feature Set Details
details = feature_set_broker.feature_set_details(0)
data_details = data_details_markdown.create("feature_set_details", details)

# Create a violin plot of all the numeric columns in the Feature Set
violin = distribution_plots.create(
    "feature_set_violin_plot",
    smart_sample_rows,
    plot_type="violin",
    figure_args={
        "box_visible": True,
        "meanline_visible": True,
        "showlegend": False,
        "points": "all",
        "spanmode": "hard",
    },
    max_plots=48,
)

# Create a correlation matrix of all the numeric columns in the Data Source
corr_df = corr_df_from_artifact_info(details)
corr_matrix = heatmap.create("feature_set_correlation_matrix", corr_df)

# Grab outlier rows and create a scatter plot
outlier_rows = feature_set_broker.feature_set_outliers(0)
outlier_plot = scatter_plot.create("outlier_plot", outlier_rows, "Clusters")


# Create our components
components = {
    "feature_sets_table": feature_sets_table,
    "feature_set_sample_rows": feature_set_sample_rows,
    "feature_set_details": data_details,
    "violin_plot": violin,
    "correlation_matrix": corr_matrix,
    "outlier_plot": outlier_plot,
}

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()

# Refresh our data timer
callbacks.refresh_data_timer(app)

# Periodic update to the data sources summary table
callbacks.update_feature_sets_table(app, feature_set_broker)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "feature_sets_table")
callbacks.update_feature_set_details(app, feature_set_broker)
callbacks.update_feature_set_sample_rows(app, feature_set_broker)
callbacks.update_violin_plots(app, feature_set_broker)
callbacks.update_correlation_matrix(app, feature_set_broker)

# Callbacks for selections
callbacks.violin_plot_selection(app)
callbacks.reorder_sample_rows(app)
callbacks.correlation_matrix_selection(app)
