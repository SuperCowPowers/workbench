"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import Dash
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.web_components import (
    table,
    data_details_markdown,
    distribution_plots,
    scatter_plot,
)
from sageworks.views.feature_set_web_view import FeatureSetWebView

# Local Imports
from layout import feature_sets_layout
import callbacks

app = Dash(
    title="SageWorks: Anomaly Inspector",
    external_stylesheets=[dbc.themes.DARKLY],
)

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

# Grab sample rows from the first data source
anomalous_rows = feature_set_broker.feature_set_anomalies(0)
feature_set_anomalies_rows = table.create(
    "feature_set_anomalies_rows",
    anomalous_rows,
    header_color="rgb(60, 60, 100)",
    max_height="400px",
)

# Data Source Details
details = feature_set_broker.feature_set_details(0)
data_details = data_details_markdown.create("feature_set_details", details)

# Create a box plot of all the numeric columns in the sample rows
smart_sample_rows = feature_set_broker.feature_set_smart_sample(0)
violin = distribution_plots.create(
    "feature_set_violin_plot",
    smart_sample_rows,
    plot_type="violin",
    figure_args={
        "box_visible": True,
        "meanline_visible": True,
        "showlegend": False,
        "points": "all",
    },
    max_plots=48,
)

# Create the anomaly cluster plot
cluster_plot = scatter_plot.create("anomaly_scatter_plot", anomalous_rows)

# Create our components
components = {
    "feature_sets_table": feature_sets_table,
    "feature_set_anomalies_rows": feature_set_anomalies_rows,
    "anomaly_scatter_plot": cluster_plot,
    "feature_set_details": data_details,
    "violin_plot": violin,
}

# Set up our layout (Dash looks for a var called layout)
app.layout = feature_sets_layout(**components)

# Refresh our data timer
callbacks.refresh_data_timer(app)

# Periodic update to the data sources summary table
callbacks.update_feature_sets_table(app, feature_set_broker)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "feature_sets_table")
callbacks.update_feature_set_details(app, feature_set_broker)
callbacks.update_feature_set_anomalies_rows(app, feature_set_broker)
callbacks.update_cluster_plot(app, feature_set_broker)
callbacks.update_violin_plots(app, feature_set_broker)

if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    # app.run_server(host="0.0.0.0", port=8080, debug=True)
    app.run_server(host="0.0.0.0", port=8080, debug=True)
