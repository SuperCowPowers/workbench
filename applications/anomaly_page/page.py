"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import Dash, register_page
import dash
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
import plotly.express as px

# SageWorks Imports
from sageworks.web_components import table, data_and_feature_details, vertical_distribution_plots, scatter_plot
from sageworks.views.feature_set_web_view import FeatureSetWebView
from sageworks.artifacts.feature_sets.feature_set import FeatureSet

# Local Imports
from .layout import anomaly_layout
from . import callbacks


app = Dash(
    title="SageWorks Anomaly",
    external_stylesheets=[dbc.themes.DARKLY],
)

register_page(__name__, path="/anomaly", name="SageWorks - anomaly")


# Put the components into 'dark' mode
load_figure_template("darkly")


# Grab sample rows from the first data source
anomaly_rows = FeatureSet("abalone_feature_set").anomalies()
anomaly_rows.sort_values(by=["cluster"], inplace=True)
anomaly_datatable = table.create(
    "anomaly_datatable",
    anomaly_rows,
    header_color="rgb(60, 60, 100)",
    max_height="400px",
    row_select="single"
)

clusters = anomaly_rows["cluster"].unique()
color_map = px.colors.qualitative.Plotly
clusters_color_map = {cluster: color_map[i] for i, cluster in enumerate(clusters)}

anomaly_datatable.style_data_conditional = [
    {
        "if": {"filter_query": "{{cluster}} ={}".format(cluster)},
        "backgroundColor": "{}".format(color),
        "color": "black",
        "border": "1px grey solid"
    } for cluster, color in clusters_color_map.items()
]

# Create a box plot of all the numeric columns in the sample rows
violin = vertical_distribution_plots.create("anomaly_violin_plot",
                                    anomaly_rows, 
                                    plot_type="violin",
                                    figure_args={"box_visible": True, "meanline_visible": True, "showlegend": False, "points": "all"},
                                    max_plots=48)

scatter = scatter_plot.create("anomaly_scatter_plot")

# Create our components
components = {
    "anomaly_table": anomaly_datatable,
    "scatter_plot": scatter,
    "violin_plot": violin,
}

# Set up our layout (Dash looks for a var called layout)
app.layout = anomaly_layout(**components)

# Refresh our data timer
callbacks.refresh_data_timer(app)

# Callbacks for when a data source is selected
# callbacks.table_row_select(app, "feature_sets_table")

# callbacks.update_violin_plots(app, feature_set_broker)


if __name__ == "__main__":
    """Run our web application in TEST mode"""
    # Note: This 'main' is purely for running/testing locally
    # app.run_server(host="0.0.0.0", port=8080, debug=True)
    app.run_server(host="0.0.0.0", port=8080, debug=True)

