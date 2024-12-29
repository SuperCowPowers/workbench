"""Callbacks for the Compound Explorer Application"""

from dash import Input, Output, callback
from dash.exceptions import PreventUpdate


# Workbench Imports
from workbench.api import FeatureSet
from workbench.web_interface.components.plugins import scatter_plot, molecule_viewer


# Set up the scatter plot callbacks
def scatter_plot_callbacks(my_scatter_plot: scatter_plot.ScatterPlot):

    # First we'll register internal callbacks for the scatter plot
    my_scatter_plot.register_internal_callbacks()

    # Now we'll set up the scatter callbacks
    @callback(
        # We can use the properties of the scatter plot to get the output properties
        [Output(component_id, prop) for component_id, prop in my_scatter_plot.properties],
        [Input("update-button", "n_clicks")],
    )
    def _scatter_plot_callbacks(_n_clicks):
        # Check for no selected rows
        # if not selected_rows or selected_rows[0] is None:
        #    raise PreventUpdate

        # Get the selected row data and grab the uuid
        # selected_row_data = selected_rows[0]
        # object_uuid = selected_row_data["uuid"]

        # Create the FeatureSet object and pull a dataframe
        df = FeatureSet("aqsol_features").pull_dataframe()

        # Update all the properties for the scatter plot
        props = my_scatter_plot.update_properties(df, hover_columns=["id"], custom_data=["id", "smiles"])

        # Return the updated properties
        return props


def molecule_view_callbacks(my_molecule_view: molecule_viewer.MoleculeViewer):
    @callback(
        [Output(component_id, prop) for component_id, prop in my_molecule_view.properties],
        Input("compound_scatter_plot-graph", "hoverData"),
    )
    def _molecule_view_callbacks(compound_data):

        # Sanity Check the Compound Data
        # print(compound_data)
        if compound_data is None:
            raise PreventUpdate
        custom_data_list = compound_data.get("points")[0].get("customdata")
        if custom_data_list is None:
            raise PreventUpdate

        # Put compound data in a dictionary
        compound_data = {
            "compound_id": custom_data_list[0],
            "smiles": custom_data_list[1],
        }

        # Update the properties for the molecule viewer
        props = my_molecule_view.update_properties(**compound_data)
        return props
