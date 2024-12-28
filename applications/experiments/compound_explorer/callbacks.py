"""Callbacks for the FeatureSets Subpage Web User Interface"""

import dash
from dash import html, Input, Output, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.api import FeatureSet
from workbench.web_interface.components.plugins import scatter_plot
from workbench.utils.chem_utils import img_from_smiles

custom_data_fields = ["id", "molwt", "smiles"]


# Set up the scatter plot callbacks
def scatter_plot_callbacks(scatter_plot: scatter_plot.ScatterPlot):

    # First we'll register internal callbacks for the scatter plot
    scatter_plot.register_internal_callbacks()

    # Now we'll set up the scatter callbacks
    @callback(
        # We can use the properties of the scatter plot to get the output properties
        [Output(component_id, prop) for component_id, prop in scatter_plot.properties],
        [Input("update-button", "n_clicks")],
    )
    def _scatter_plot_callbacks(n_clicks):
        # Check for no selected rows
        # if not selected_rows or selected_rows[0] is None:
        #    raise PreventUpdate

        # Get the selected row data and grab the uuid
        # selected_row_data = selected_rows[0]
        # object_uuid = selected_row_data["uuid"]

        # Create the FeatureSet object and pull a dataframe
        df = FeatureSet("aqsol_features").pull_dataframe()

        # Update all the properties for the scatter plot
        props = scatter_plot.update_properties(df, hover_columns=["id"], custom_data=custom_data_fields)

        # Return the updated properties
        return props


def update_compound_diagram():
    @callback(
        Output("compound_diagram", "children"),
        Input("compound_scatter_plot-graph", "hoverData"),
    )
    def diagram_update(compound_data):

        # Sanity Check the Compound Data
        print(compound_data)
        if compound_data is None:
            raise PreventUpdate
        custom_data_list = compound_data.get("points")[0].get("customdata")
        if custom_data_list is None:
            raise PreventUpdate

        # Put compound data in a dictionary
        compound_data = dict(zip(custom_data_fields, custom_data_list))
        id = compound_data["id"]
        mol_weight = compound_data["molwt"]
        smiles = compound_data["smiles"]
        print(f"Smiles Data: {smiles}")

        # Create the Molecule Image
        img = img_from_smiles(smiles)

        # Sanity Check
        if img is None:
            print("**** Could not generate an image ****")
            return dash.no_update

        # New 'Children' for the Compound Diagram
        children = [
            dbc.Row(
                html.H5(f"Compound: {id}"),
                style={"padding": "0px 0px 0px 0px"},
            ),
            dbc.Row(
                html.Img(src=img),
                style={"padding": "0px 0px 0px 0px"},
            ),
            dbc.Row(
                html.H5(f"Molecular Weight: {mol_weight}"),
                style={"padding": "0px 0px 0px 0px"},
            ),
        ]

        # Return the children of the Compound Diagram
        return children
