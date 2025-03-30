"""Callbacks for the FeatureSets Subpage Web User Interface"""

import dash
from dash import callback, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import logging
from urllib.parse import urlparse, parse_qs

# Workbench Imports
from workbench.web_interface.page_views.feature_sets_page_view import FeatureSetsPageView
from workbench.web_interface.components import violin_plots, correlation_matrix
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.web_interface.components.plugins.data_details import DataDetails
from workbench.cached.cached_feature_set import CachedFeatureSet

# Set up logging
log = logging.getLogger("workbench")


# Cheese Sauce
smart_sample_rows = []


def on_page_load():
    @callback(
        Output("feature_sets_table", "selectedRows"),
        Output("feature_sets_page_loaded", "data"),
        Input("url", "href"),
        Input("feature_sets_table", "rowData"),
        State("feature_sets_page_loaded", "data"),
        prevent_initial_call=True,
    )
    def _on_page_load(href, row_data, page_already_loaded):
        if page_already_loaded:
            raise PreventUpdate

        if not href or not row_data:
            raise PreventUpdate

        parsed = urlparse(href)
        if parsed.path != "/feature_sets":
            raise PreventUpdate

        selected_uuid = parse_qs(parsed.query).get("uuid", [None])[0]
        if not selected_uuid:
            return [row_data[0]], True

        for row in row_data:
            if row.get("uuid") == selected_uuid:
                return [row], True

        raise PreventUpdate


def feature_sets_refresh(page_view: FeatureSetsPageView, fs_table: AGTable):
    @callback(
        [Output(component_id, prop) for component_id, prop in fs_table.properties],
        Input("feature_sets_refresh", "n_intervals"),
    )
    def _feature_sets_refresh(_n):
        """Return the table data for the FeatureSets Table"""
        page_view.refresh()
        feature_sets = page_view.feature_sets()
        feature_sets["uuid"] = feature_sets["Feature Group"]
        feature_sets["id"] = range(len(feature_sets))
        return fs_table.update_properties(feature_sets)


# Updates the feature set details and correlation matrix when a new FeatureSet is selected
def update_feature_set_details(feature_details: DataDetails):
    @callback(
        [
            Output("feature_set_details-header", "children"),
            Output("feature_set_details-details", "children"),
            Output("feature_set_correlation_matrix", "figure", allow_duplicate=True),
        ],
        Input("feature_sets_table", "selectedRows"),
        prevent_initial_call=True,
    )
    def generate_feature_set_markdown(selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update

        # Get the selected row data and grab the uuid
        selected_row_data = selected_rows[0]
        feature_set_uuid = selected_row_data["uuid"]
        print(f"FeatureSet UUID: {feature_set_uuid}")
        fs = CachedFeatureSet(feature_set_uuid)

        # FeatureSet Details Markdown component
        header, feature_details_markdown = feature_details.update_properties(fs)

        # Generate a new correlation matrix figure
        corr_figure = correlation_matrix.CorrelationMatrix().update_properties(fs.details())

        # Return the details/markdown for these data details
        return [header, feature_details_markdown, corr_figure]


def update_feature_set_sample_rows(samples_table: AGTable):
    @callback(
        [
            Output("feature_sample_rows_header", "children"),
            Output("feature_set_sample_rows", "columnDefs"),
            Output("feature_set_sample_rows", "rowData"),
            Output("feature_set_violin_plot", "figure", allow_duplicate=True),
        ],
        Input("feature_sets_table", "selectedRows"),
        prevent_initial_call=True,
    )
    def smart_sample_rows_update(selected_rows):
        global smart_sample_rows
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update

        # Get the selected row data and grab the uuid
        selected_row_data = selected_rows[0]
        feature_set_uuid = selected_row_data["uuid"]
        print(f"FeatureSet UUID: {feature_set_uuid}")
        fs = CachedFeatureSet(feature_set_uuid)

        print("Calling FeatureSet Sample Rows...")
        smart_sample_rows = fs.smart_sample()

        # Header Text
        header = f"Sample/Outlier Rows: {feature_set_uuid}"

        # Grab column definitions and row data from our Samples Table
        [column_defs, _, _] = samples_table.update_properties(smart_sample_rows)

        # Update the Violin Plot with the new smart sample rows
        violin_figure = violin_plots.ViolinPlots().update_properties(
            smart_sample_rows,
            figure_args={
                "box_visible": True,
                "meanline_visible": True,
                "showlegend": False,
                "points": "all",
                "spanmode": "hard",
            },
        )

        # Return the header, columns, style_cell, and the data
        return [header, column_defs, smart_sample_rows.to_dict("records"), violin_figure]


#
# The following callbacks are for selections
#


def violin_plot_selection():
    """A selection has occurred on the Violin Plots so highlight the selected points on the plot,
    and send the updated figure to the client"""

    @callback(
        Output("feature_set_violin_plot", "figure", allow_duplicate=True),
        Input("feature_set_violin_plot", "selectedData"),
        State("feature_set_violin_plot", "figure"),
        prevent_initial_call=True,
    )
    def update_figure(selected_data, current_figure):
        # Get the selected indices
        if selected_data is None:
            selected_indices = []
        else:
            selected_indices = [point["pointIndex"] for point in selected_data["points"]]
        log.info("Selected Indices")
        log.info(selected_indices)

        # Create a figure object so that we can use nice methods like update_traces
        figure = go.Figure(current_figure)

        # Update the selected points
        figure.update_traces(selectedpoints=selected_indices, selector=dict(type="violin"))
        return figure


def get_selection_indices(click_data, df: pd.DataFrame):
    """Get the selection indices from the columns clicked on in the correlation matrix"""

    # First lets get the column names and the correlation
    first_column = click_data["points"][0]["y"].split(":")[0]
    second_column = click_data["points"][0]["x"].split(":")[0]
    correlation = click_data["points"][0]["z"]
    log.info(f"First Column: {first_column}")
    log.info(f"Second Column: {second_column}")
    log.info(f"Correlation: {correlation}")

    # Now grab the indexes for the top 10 value from the first column
    selection_indices = set(df[first_column].nlargest(10).index.tolist())

    # If the correlation is positive, then grab the top 10 values from the
    # second column otherwise grab the bottom 10 values
    if correlation > 0:
        selection_indices = selection_indices.union(set(df[second_column].nlargest(10).index.tolist()))
    elif correlation == 0:
        selection_indices = []
    else:
        selection_indices = selection_indices.union(set(df[second_column].nsmallest(10).index.tolist()))

    # Return the selected indices
    return list(selection_indices)


def select_row_column(figure, click_data):
    """Select a row and column in the correlation matrix based on click data and the dataframe"""

    # Get the columns index from the click_data
    first_column_index = int(click_data["points"][0]["x"].split(":")[1])
    second_column_index = int(click_data["points"][0]["y"].split(":")[1])
    log.info(f"First Column Index: {first_column_index}")
    log.info(f"Second Column Index: {second_column_index}")

    # Clear any existing shapes (highlights)
    figure["layout"]["shapes"] = ()

    # Add a rectangle shape to outline the cell
    figure.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=first_column_index - 0.5,
        y0=second_column_index - 0.5,
        x1=first_column_index + 0.5,
        y1=second_column_index + 0.5,
        line=dict(color="White"),
    )


def correlation_matrix_selection():
    """A selection has occurred on the Correlation Matrix so highlight the selected box, and also update
    the selections in the violin plot"""

    @callback(
        [
            Output("feature_set_correlation_matrix", "figure", allow_duplicate=True),
            Output("feature_set_violin_plot", "figure", allow_duplicate=True),
        ],
        Input("feature_set_correlation_matrix", "clickData"),
        State("feature_set_correlation_matrix", "figure"),
        State("feature_set_violin_plot", "figure"),
        State("feature_set_sample_rows", "rowData"),
        prevent_initial_call=True,
    )
    def update_figure(click_data, corr_figure, violin_figure, sample_rows):
        # Convert the sample rows to a DataFrame
        sample_rows = pd.DataFrame(sample_rows)

        # Create a selection box in the correlation matrix
        corr_figure = go.Figure(corr_figure)

        # Add a rectangle shape to outline the cell
        select_row_column(corr_figure, click_data)

        # Update the selected points in the violin figure
        if click_data:
            selected_indices = get_selection_indices(click_data, sample_rows)
        else:
            selected_indices = []
        violin_figure = go.Figure(violin_figure)
        violin_figure.update_traces(selectedpoints=selected_indices, selector=dict(type="violin"))
        return [corr_figure, violin_figure]


def reorder_sample_rows():
    """A selection has occurred on the Violin Plots so highlight the selected points on the plot,
    regenerate the figure"""

    @callback(
        Output("feature_set_sample_rows", "rowData", allow_duplicate=True),
        Input("feature_set_violin_plot", "selectedData"),
        prevent_initial_call=True,
    )
    def reorder_table(selected_data):
        # Convert the current table data back to a DataFrame

        # Get the selected indices from your plot selection
        if selected_data is None or smart_sample_rows is None:
            return dash.no_update
        selected_indices = [point["pointIndex"] for point in selected_data["points"]]

        # Separate the selected rows and the rest of the rows
        selected_rows = smart_sample_rows.iloc[selected_indices]
        rest_of_rows = smart_sample_rows.drop(selected_indices)

        # Concatenate them to put the selected rows at the top
        new_df = pd.concat([selected_rows, rest_of_rows], ignore_index=True)

        # Return the new DataFrame as a dictionary
        return new_df.to_dict("records")
