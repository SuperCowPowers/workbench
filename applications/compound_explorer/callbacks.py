"""Callbacks for the Compound Explorer Application"""

from dash import Input, Output, callback, html, no_update

# Workbench Imports
from workbench.api import FeatureSet  # noqa: F401
from workbench.api import Compound, df_store
from workbench.web_interface.components.plugins import scatter_plot, compound_details


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

        # Create the FeatureSet object and pull a dataframe
        # df = FeatureSet("aqsol_features").pull_dataframe()

        # Load our preprocessed tox21 training data
        df = df_store.DFStore().get("/datasets/chem_info/tox21")

        # Generate the molecules (they can't be serialized)
        # df['molecule'] = df['smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles))

        # Update all the properties for the scatter plot
        props = my_scatter_plot.update_properties(
            df,
            hover_columns=["id"],
            custom_data=["id", "smiles", "tags"],
            suppress_hover_display=True,
            x="x",
            y="y",
            color="toxic_any",
        )

        # Return the updated properties
        return props


def molecule_view_callbacks(my_compound_view: compound_details.CompoundDetails):
    @callback(
        [Output(component_id, prop) for component_id, prop in my_compound_view.properties]
        + [
            Output("hover-tooltip", "show"),
            Output("hover-tooltip", "bbox"),
            Output("hover-tooltip", "children"),
        ],
        Input("compound_scatter_plot-graph", "hoverData"),
    )
    def _molecule_view_callbacks(hover_data):

        # Sanity Check that we get the data we need to update the molecule viewer
        if hover_data is None:
            return no_update, no_update, no_update, False, no_update, no_update
        custom_data_list = hover_data.get("points")[0].get("customdata")
        if custom_data_list is None:
            return no_update, no_update, no_update, False, no_update, no_update

        # Spin up the compound object
        compound_id = custom_data_list[0]
        smiles = custom_data_list[1]
        tags = custom_data_list[2]
        compound = Compound(compound_id)
        compound.smiles = smiles
        compound.tags = tags

        # Update the properties for the molecule viewer
        [header_text, img, details] = my_compound_view.update_properties(compound)

        # Set up the outputs for the hover tooltip
        show = True
        bbox = hover_data["points"][0]["bbox"]
        bbox["x0"] += 150
        bbox["x1"] += 150
        children = [
            html.Img(
                src=img,
                style={"padding": "0px", "margin": "0px"},
                className="custom-tooltip",
                width="300",
                height="200",
            )
        ]
        return [header_text, img, details, show, bbox, children]
