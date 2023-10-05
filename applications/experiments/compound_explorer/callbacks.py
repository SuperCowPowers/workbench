"""Callbacks for the FeatureSets Subpage Web User Interface"""
from datetime import datetime
import dash
from dash import Dash, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.views.data_source_web_view import DataSourceWebView
from sageworks.web_components import (
    table,
    compound_details,
    violin_plots,
    scatter_plot,
)

# FIXME
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.rdMolDraw2D import SetDarkMode

first_m = Chem.MolFromSmiles("O=C1Nc2cccc3cccc1c23")
dos = Draw.MolDrawOptions()
SetDarkMode(dos)
dos.setBackgroundColour((0, 0, 0, 0))


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


# Updates the data source details when a row is selected in the summary table
def update_data_source_details(app: Dash, data_source_web_view: DataSourceWebView):
    @app.callback(
        [
            Output("data_source_details_header", "children"),
            Output("data_source_details", "children"),
        ],
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
    )
    def generate_new_markdown(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling DataSource Details...")
        data_source_details = data_source_web_view.data_source_details(selected_rows[0])
        data_source_details_markdown = compound_details.create_markdown(data_source_details)

        # Name of the data source for the Header
        data_source_name = data_source_web_view.data_source_name(selected_rows[0])
        header = f"Dataset: {data_source_name}"

        # Return the details/markdown for these data details
        return [header, data_source_details_markdown]


def update_compound_rows(app: Dash, data_source_web_view: DataSourceWebView):
    @app.callback(
        [
            Output("data_source_rows_header", "children"),
            Output("compound_rows", "columns"),
            Output("compound_rows", "data"),
        ],
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def sample_rows_update(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling DataSource Sample Rows...")
        sample_rows = data_source_web_view.data_source_outliers(selected_rows[0])

        # To select rows we need to set up an (0->N) ID for each row
        sample_rows["id"] = range(len(sample_rows))

        # Name of the data source
        data_source_name = data_source_web_view.data_source_name(selected_rows[0])
        header = f"{data_source_name}: compounds"

        # The columns need to be in a special format for the DataTable
        column_setup_list = table.Table().column_setup(sample_rows)

        # Return the columns and the data
        return [header, column_setup_list, sample_rows.to_dict("records")]


def update_compound_diagram(app: Dash):
    @app.callback(
        Output("compound_diagram", "children"),
        Input("compound_rows", "derived_viewport_selected_rows"),
        State("compound_rows", "derived_viewport_data"),
        prevent_initial_call=True,
    )
    def diagram_update(selected_rows, compound_data):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        print("Calling Compound Diagram Update...")
        compound_name = compound_data[selected_rows[0]].get("name", "Unknown")
        smiles = compound_data[selected_rows[0]].get("smiles", "Unknown")
        mol_weight = compound_data[selected_rows[0]].get("molwt", 0.0)
        print(f"Smiles Data: {smiles}")
        m = Chem.MolFromSmiles(smiles)

        # Sanity Check the Molecule
        if m is None:
            print("**** Molecule is None ****")
            return dash.no_update

        # New 'Children' for the Compound Diagram
        children = [
            dbc.Row(
                html.H5(compound_name),
                style={"padding": "0px 0px 0px 0px"},
            ),
            dbc.Row(
                html.Img(src=Draw.MolToImage(m, options=dos, size=(300, 300)), style={"height": "300", "width": "300"}),
                style={"padding": "0px 0px 0px 0px"},
            ),
            dbc.Row(
                html.H5(f"Molecular Weight: {mol_weight}"),
                style={"padding": "0px 0px 0px 0px"},
            ),
        ]

        # Return the children of the Compound Diagram
        return children


def update_cluster_plot(app: Dash, data_source_web_view: DataSourceWebView):
    """Updates the Cluster Plot when a new data source is selected"""

    @app.callback(
        Output("compound_scatter_plot", "figure"),
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_cluster_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        outlier_rows = data_source_web_view.data_source_outliers(selected_rows[0])
        return scatter_plot.create_figure(outlier_rows)


def update_violin_plots(app: Dash, data_source_web_view: DataSourceWebView):
    """Updates the Violin Plots when a new feature set is selected"""

    @app.callback(
        Output("data_source_violin_plot", "figure"),
        Input("data_sources_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_violin_plot(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        smart_sample_rows = data_source_web_view.data_source_smart_sample(selected_rows[0])
        return violin_plots.ViolinPlots().generate_component_figure(
            smart_sample_rows,
            figure_args={
                "box_visible": True,
                "meanline_visible": True,
                "showlegend": False,
                "points": "all",
            },
        )
