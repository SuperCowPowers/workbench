"""Callbacks for the FeatureSets Subpage Web User Interface"""
from datetime import datetime
import dash
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.feature_set_web_view import FeatureSetWebView
from sageworks.web_components import data_and_feature_details, vertical_distribution_plots
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


def refresh_data_timer(app: Dash):
    @app.callback(
        Output("last-updated-anomaly-page", "children"),
        Input("anomaly-updater", "n_intervals"),
    )
    def time_updated(_n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")

def update_feature_sets_table(app: Dash, feature_set_broker: FeatureSetWebView):
    @app.callback(Output("feature_sets_table", "data"), Input("anomaly-updater", "n_intervals"))
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


def update_anomaly_table(app: Dash):
    @app.callback(
        [
            Output("anomaly-table-header", "children"),
            Output("anomaly_table", "columns"),
            Output("anomaly_table", "data"),
        ],
        Input("feature_sets_table", "derived_viewport_selected_row_ids"),
        State("feature_sets_table", "data"),
        # prevent_initial_call=True,
    )
    def sample_rows_update(selected_rows, feature_sets_data):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return no_update, no_update, no_update
       
        feature_set_uuid = feature_sets_data[selected_rows[0]]["uuid"]
        anomaly_rows = FeatureSet(feature_set_uuid).anomalies()
        anomaly_rows.sort_values(by=["cluster"], inplace=True)
        
        header = f"Anomalies from: {feature_set_uuid}"

        # # The columns need to be in a special format for the DataTable
        # column_setup = [{"name": c, "id": c, "presentation": "input"} for c in sample_rows.columns]

        # Return the columns and the data
        return header, no_update, anomaly_rows.to_dict("records")


def update_violin_plots(app: Dash, feature_set_web_view: FeatureSetWebView):
    """Updates the Violin Plots when a new data source is selected"""

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
