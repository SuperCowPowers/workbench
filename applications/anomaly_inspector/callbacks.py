"""Callbacks for the FeatureSets Subpage Web User Interface"""
from datetime import datetime
from typing import Dict, List
import dash
from dash import Dash
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.feature_set_web_view import FeatureSetWebView
from sageworks.web_components import data_and_feature_details, vertical_distribution_plots, scatter_plot


def refresh_data_timer(app: Dash):
    @app.callback(
        Output("last-updated-feature-sets", "children"),
        Input("feature-sets-updater", "n_intervals"),
    )
    def time_updated(_n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_feature_sets_table(app: Dash, feature_set_broker: FeatureSetWebView):
    @app.callback(Output("feature_sets_table", "data"), Input("feature-sets-updater", "n_intervals"))
    def feature_sets_update(_n):
        """Return the table data as a dictionary"""
        feature_set_broker.refresh()
        feature_set_rows = feature_set_broker.feature_sets_summary()
        feature_set_rows["id"] = feature_set_rows.index
        return feature_set_rows.to_dict("records")


# Highlights the selected row in the table
def table_row_select(app: Dash, table_name: str):
    @app.callback(
        Output(table_name, "style_data_conditional"),
        Input(table_name, "derived_viewport_selected_row_ids"),
    )
    def style_selected_rows(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        row_style = [
            {
                "if": {"filter_query": "{{id}} ={}".format(i)},
                "backgroundColor": "rgb(80, 80, 80)",
            }
            for i in selected_rows
        ]
        return row_style


# Updates the data source details when a row is selected in the summary
def update_feature_set_details(app: Dash, feature_set_web_view: FeatureSetWebView):
    @app.callback(
        [Output("feature_details_header", "children"), Output("feature_set_details", "children")],
        Input("feature_sets_table", "derived_viewport_selected_row_ids"),
    )
    def generate_new_markdown(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling FeatureSet Details...")
        feature_details = feature_set_web_view.feature_set_details(selected_rows[0])
        feature_details_markdown = data_and_feature_details.create_markdown(feature_details)

        # Name of the data source for the Header
        feature_set_name = feature_set_web_view.feature_set_name(selected_rows[0])
        header = f"Details: {feature_set_name}"

        # Return the details/markdown for these data details
        return [header, feature_details_markdown]


def update_feature_set_anomalies_rows(app: Dash, feature_set_web_view: FeatureSetWebView, color_list: List):
    @app.callback(
        [
            Output("feature_sample_rows_header", "children"),
            Output("feature_set_anomalies_rows", "columns"),
            Output("feature_set_anomalies_rows", "data"),
            Output("feature_set_anomalies_rows", "style_data_conditional"),
        ],
        Input("feature_sets_table", "derived_viewport_selected_row_ids"),
        State("feature-sets-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def sample_rows_update(selected_rows, _n):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None or _n == 0:
            return dash.no_update
        print("Calling FeatureSet Sample Rows...")
        anomalous_rows = feature_set_web_view.feature_set_anomalies(selected_rows[0])
        anomalous_rows.sort_values(by=["cluster"], inplace=True)

        # Name of the data source
        feature_set_name = feature_set_web_view.feature_set_name(selected_rows[0])
        header = f"Anomalous Rows: {feature_set_name}"

        # The columns need to be in a special format for the DataTable
        column_setup = [{"name": c, "id": c, "presentation": "input"} for c in anomalous_rows.columns]

        # Set the color row based on the cluster value
        clusters = anomalous_rows["cluster"].unique()
        style_data_conditional = [
            {
                "if": {"filter_query": "{{cluster}} ={}".format(cluster)},
                "backgroundColor": "{}".format(color_list[cluster]),
                "color": "black",
                "border": "1px grey solid"
            } for cluster in clusters
        ]

        # Return the columns and the data
        return [header, column_setup, anomalous_rows.to_dict("records"), style_data_conditional]


def update_cluster_plot(app: Dash, feature_set_web_view: FeatureSetWebView, color_list: List):
    """Updates the Cluster Plot when a new feature set is selected"""

    @app.callback(
        Output("anomaly_scatter_plot", "figure"),
        Input("feature_sets_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_cluster_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling FeatureSet Sample Rows Refresh...")
        anomalous_rows = feature_set_web_view.feature_set_anomalies(selected_rows[0])
        anomalous_rows.sort_values(by=["cluster"], inplace=True)
        clusters = anomalous_rows["cluster"].unique()
        color_discrete_map = {str(cluster): str(color_list[i]) for i, cluster in enumerate(clusters)}
        return scatter_plot.create_figure(anomalous_rows, color_discrete_map)


def update_violin_plots(app: Dash, feature_set_web_view: FeatureSetWebView):
    """Updates the Violin Plots when a new feature set is selected"""

    @app.callback(
        Output("feature_set_violin_plot", "figure"),
        Input("feature_sets_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_violin_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling FeatureSet Sample Rows Refresh...")
        smart_sample_rows = feature_set_web_view.feature_set_smart_sample(selected_rows[0])
        return vertical_distribution_plots.create_figure(smart_sample_rows,
                                           plot_type="violin",
                                           figure_args={"box_visible": True, "meanline_visible": True, "showlegend": False, "points": "all"},
                                           max_plots=48)
