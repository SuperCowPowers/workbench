"""FeatureSets Callbacks: Callback within the DataSources Web User Interface"""
from datetime import datetime
import dash
from dash import Dash
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.web_data_source_view import WebDataSourceView
from sageworks.web_components import violin_plot


def update_last_updated(app: Dash):
    @app.callback(Output("last-updated-data-sources", "children"), Input("data-sources-updater", "n_intervals"))
    def time_updated(n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_data_sources_summary(app: Dash, web_data_source_view: WebDataSourceView):
    @app.callback(
        [Output("data_sources_summary", "data"), Output("data_sources_summary", "selected_rows")],
        Input("data-sources-updater", "n_intervals"),
        State("data_sources_summary", "selected_rows")
    )
    def data_sources_update(n, selected_rows):
        print("Calling DataSources Summary Refresh...")
        web_data_source_view.refresh()
        data_sources = web_data_source_view.data_sources_summary()
        return [data_sources.to_dict("records"), selected_rows]


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


def update_sample_rows_header(app: Dash):
    @app.callback(
        Output("sample_rows_header", "children"),
        Input("data_sources_summary", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True
    )
    def update_sample_header(selected_rows):
        return "Sampled Rows: Updating..."


def update_data_source_sample_rows(app: Dash, web_data_source_view: WebDataSourceView):
    @app.callback(
        [Output("sample_rows_header", "children", allow_duplicate=True),
         Output("data_source_sample_rows", "columns"),
         Output("data_source_sample_rows", "data")],
        Input("data_sources_summary", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True
    )
    def sample_rows_update(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling DataSource Sample Rows Refresh...")
        sample_rows = web_data_source_view.data_source_sample(selected_rows[0])

        # Name of the data source
        data_source_name = web_data_source_view.data_source_name(selected_rows[0])
        header = f"Sampled Rows: {data_source_name}"

        # The columns need to be in a special format for the DataTable
        column_setup = [{"name": c, "id": c, "presentation": "input"} for c in sample_rows.columns]

        # Return the columns and the data
        return [header, column_setup, sample_rows.to_dict("records")]


def update_violin_plots(app: Dash, web_data_source_view: WebDataSourceView):
    """Updates the Violin Plots when a new data source is selected"""
    @app.callback(
        Output("violin_plot", "figure"),
        Input("data_sources_summary", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True
    )
    def generate_new_violin_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling DataSource Sample Rows Refresh...")
        sample_rows = web_data_source_view.data_source_sample(selected_rows[0])
        return violin_plot.create_figure(sample_rows)
