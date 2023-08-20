"""FeatureSets Callbacks: Callback within the DataSources Web User Interface"""
from datetime import datetime
import dash
from dash import Dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# SageWorks Imports
from sageworks.views.data_source_web_view import DataSourceWebView
from sageworks.web_components import data_details_markdown, distribution_plots, heatmap, scatter_plot
from sageworks.utils.pandas_utils import corr_df_from_artifact_info


def refresh_data_timer(app: Dash):
    @app.callback(
        Output("last-updated-data-sources", "children"),
        Input("data-sources-updater", "n_intervals"),
    )
    def time_updated(_n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_data_sources_table(app: Dash, data_source_broker: DataSourceWebView):
    @app.callback(
        Output("data_sources_table", "data"),
        Input("data-sources-updater", "n_intervals"),
    )
    def data_sources_update(_n):
        """Return the table data as a dictionary"""
        data_source_broker.refresh()
        data_source_rows = data_source_broker.data_sources_summary()
        data_source_rows["id"] = data_source_rows.index
        return data_source_rows.to_dict("records")


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


# Updates the data source details when a new DataSource is selected
def update_data_source_details(app: Dash, data_source_web_view: DataSourceWebView):
    @app.callback(
        [
            Output("data_details_header", "children"),
            Output("data_source_details", "children"),
        ],
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
    )
    def generate_new_markdown(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling DataSource Details...")
        data_details = data_source_web_view.data_source_details(selected_rows[0])
        details_markdown = data_details_markdown.create_markdown(data_details)

        # Name of the data source for the Header
        data_source_name = data_source_web_view.data_source_name(selected_rows[0])
        header = f"Details: {data_source_name}"

        # Return the details/markdown for these data details
        return [header, details_markdown]


def update_data_source_sample_rows(app: Dash, data_source_web_view: DataSourceWebView):
    @app.callback(
        [
            Output("sample_rows_header", "children"),
            Output("data_source_sample_rows", "columns"),
            Output("data_source_sample_rows", "data"),
        ],
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def sample_rows_update(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling DataSource Sample Rows...")
        sample_rows = data_source_web_view.data_source_sample(selected_rows[0])

        # Name of the data source
        data_source_name = data_source_web_view.data_source_name(selected_rows[0])
        header = f"Sampled Rows: {data_source_name}"

        # The columns need to be in a special format for the DataTable
        column_setup = [{"name": c, "id": c, "presentation": "input"} for c in sample_rows.columns]

        # Return the columns and the data
        return [header, column_setup, sample_rows.to_dict("records")]


def update_violin_plots(app: Dash, data_source_web_view: DataSourceWebView):
    """Updates the Violin Plots when a new data source is selected"""

    @app.callback(
        Output("data_source_violin_plot", "figure", allow_duplicate=True),
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def generate_new_violin_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update

        # Get the data source smart sample rows and create the violin plot
        smart_sample_rows = data_source_web_view.data_source_smart_sample(selected_rows[0])
        return distribution_plots.create_figure(
            smart_sample_rows,
            plot_type="violin",
            figure_args={
                "box_visible": True,
                "meanline_visible": True,
                "showlegend": False,
                "points": "all",
                "spanmode": "hard"
            },
            max_plots=48,
        )


def violin_plot_selection(app: Dash, data_source_web_view: DataSourceWebView):
    """A selection has occurred on the Violin Plots so highlight the selected points on the plot,
       regenerate the figure and update the Outlier Rows (TBD)"""
    @app.callback(
        Output('data_source_violin_plot', 'figure', allow_duplicate=True),
        Input('data_source_violin_plot', 'selectedData'),
        State('data_source_violin_plot', 'figure'),
        prevent_initial_call=True,
    )
    def update_figure(selected_data, current_figure):

        # If we don't have any selected data our selection is empty
        if selected_data is None:
            return dash.no_update

        # Get the selected indices
        selected_indices = [point['pointIndex'] for point in selected_data['points']]
        print("Selected Indices")
        print(selected_indices)

        # Create a new figure that's a copy of the original with selected data highlighted
        try:
            new_figure = go.Figure(current_figure)
        except:
            return current_figure

        # Update the selected points
        new_figure.update_traces(selectedpoints=selected_indices, selector=dict(type='violin'))
        return new_figure


# Updates the correlation matrix when a new DataSource is selected
def update_correlation_matrix(app: Dash, data_source_web_view: DataSourceWebView):
    @app.callback(
        Output("corr_matrix", "figure"),
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_corr_matrix(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update

        # Get the data source smart sample rows and create the correlation matrix
        artifact_info = data_source_web_view.data_source_details(selected_rows[0])

        # Convert the data details to a pandas dataframe
        corr_df = corr_df_from_artifact_info(artifact_info)
        return heatmap.create_figure(corr_df)


# Updates the outlier plot when a new DataSource is selected
def update_outlier_plot(app: Dash, data_source_web_view: DataSourceWebView):
    """Updates the Outlier Plot when a new data source is selected"""

    @app.callback(
        Output("outlier_plot", "figure"),
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_outlier_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        outlier_rows = data_source_web_view.data_source_outliers(selected_rows[0])
        return scatter_plot.create_figure(outlier_rows)
