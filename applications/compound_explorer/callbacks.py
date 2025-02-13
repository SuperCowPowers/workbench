"""Callbacks for the Compound Explorer Application"""

from dash import Input, Output, callback, html, no_update
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.api import df_store
from workbench.api.compound import Compound
from workbench.web_interface.components.plugins import scatter_plot


# Populate the initial data for the scatter plot
def populate_scatter_plot(my_scatter_plot: scatter_plot.ScatterPlot):

    # First we'll register internal callbacks for the scatter plot
    my_scatter_plot.register_internal_callbacks()

    # This is just a 'fake' callback to get the scatter plot to load
    @callback(
        # We can use the properties of the scatter plot to get the output properties
        [Output(component_id, prop) for component_id, prop in my_scatter_plot.properties],
        [Input("invisible-go-button", "n_clicks")],
    )
    def _populate_scatter_plot(_n_clicks):

        # Load our preprocessed tox21 training data
        df = df_store.DFStore().get("/datasets/chem_info/tox21")

        # Silly tox colors (for now)
        num_elements = df["meta"].apply(lambda x: len(x["toxic_elements"]) if x["toxic_elements"] is not None else 0)
        num_groups = df["meta"].apply(lambda x: len(x["toxic_groups"]) if x["toxic_groups"] is not None else 0)
        df["tox_alert"] = num_elements + num_groups + df["toxic_any"]

        # Update all the properties for the scatter plot
        props = my_scatter_plot.update_properties(
            df,
            hover_columns=["id"],
            custom_data=["id", "smiles", "tags", "meta"],
            suppress_hover_display=True,
            x="x",
            y="y",
            color="tox_alert",
        )

        # Return the updated properties
        return props


def hover_tooltip_callbacks():
    @callback(
        [
            Output("hover-tooltip", "show"),
            Output("hover-tooltip", "bbox"),
            Output("hover-tooltip", "children"),
        ],
        Input("compound_scatter_plot-graph", "hoverData"),
    )
    def _hover_tooltip_callbacks(hover_data):

        # Sanity Check that we get the data we need to update the molecule viewer
        if hover_data is None:
            return False, no_update, no_update
        custom_data_list = hover_data.get("points")[0].get("customdata")
        if custom_data_list is None:
            return False, no_update, no_update

        # Construct a compound object (hardcoded to the hoverData format)
        compound = Compound(custom_data_list[0])
        compound.smiles = custom_data_list[1]
        compound.tags = custom_data_list[2]
        compound.meta = custom_data_list[3]

        # Generate the molecule image for the hover tooltip
        img = compound.image()

        # Set up the outputs for the hover tooltip
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
        return [True, bbox, children]


def setup_plugin_callbacks(plugins):

    # First we'll register internal callbacks for the plugins
    for plugin in plugins:
        plugin.register_internal_callbacks()

    # Now we'll set up the plugin callbacks for their main inputs (models in this case)
    @callback(
        # Aggregate plugin outputs
        [Output(component_id, prop) for p in plugins for component_id, prop in p.properties],
        Input("compound_scatter_plot-graph", "hoverData"),
    )
    def update_all_plugin_properties(hover_data):
        # Check for no hover data
        # Sanity Check that we get the data we need to update the molecule viewer
        if hover_data is None:
            raise PreventUpdate
        custom_data_list = hover_data.get("points")[0].get("customdata")
        if custom_data_list is None:
            raise PreventUpdate

        # Construct a compound object (hardcoded to the hoverData format)
        compound = Compound(custom_data_list[0])
        compound.smiles = custom_data_list[1]
        compound.tags = custom_data_list[2]
        compound.meta = custom_data_list[3]

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(compound))

        # Return all the updated properties
        return all_props
