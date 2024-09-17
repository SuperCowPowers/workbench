"""Model Data Quality Plugin Page: A SageWorks Plugin Page Interface"""

import dash
from dash import html, page_container, register_page, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# SageWorks Imports
from sageworks.api import FeatureSet
from sageworks.web_components import table
from sageworks.api.meta import Meta
from sageworks.web_components.plugins.data_details import DataDetails
from sageworks.web_components.plugins.scatter_plot import ScatterPlot


class MDQPluginPage:
    """Model Data Quality Plugin Page: A SageWorks Plugin Page Interface"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "FeatureSets: Model Data Quality"
        self.meta = Meta()

        # UI Components
        self.feature_sets_table = table.Table().create_component(
            "mdq_feature_set_table", header_color="rgb(60, 60, 60)", row_select="single", max_height=400
        )
        self.feature_set_details = DataDetails()
        self.feature_set_details_component = self.feature_set_details.create_component("mdq_feature_set_details")
        self.scatter_plot = ScatterPlot()
        self.scatter_plot_component = self.scatter_plot.create_component("mdq_scatter_plot")
        self.plugins = [self.feature_set_details, self.scatter_plot]

    def page_setup(self, app: dash.Dash):
        """Page Setup: Register the page, set the layout, register callbacks, and populate any widgets"""

        # Register this page with Dash and set up the layout
        register_page(
            __file__,
            path="/mdq_view",
            name=self.page_name,
            layout=self.page_layout(),
        )

        # Populate the table with data about the FeatureSets
        feature_sets = self.meta.feature_sets()
        feature_sets["uuid"] = feature_sets["Feature Group"]
        feature_sets["id"] = range(len(feature_sets))
        self.feature_sets_table.columns = table.Table().column_setup(feature_sets)
        self.feature_sets_table.data = feature_sets.to_dict("records")

        # Register the callbacks
        self.setup_plugin_callbacks()
        self.scatter_plot.register_internal_callbacks()

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        layout = dash.html.Div(
            children=[
                dash.html.H1(self.page_name),
                dbc.Row(self.feature_sets_table),
                dbc.Row(
                    [
                        dbc.Col(self.feature_set_details_component, width=5),
                        dbc.Col(self.scatter_plot_component, width=7),
                    ]
                ),
            ]
        )
        return layout

    def setup_plugin_callbacks(self):
        @callback(
            # Aggregate plugin outputs
            [Output(component_id, prop) for p in self.plugins for component_id, prop in p.properties],
            Input("mdq_feature_set_table", "derived_viewport_selected_row_ids"),
            State("mdq_feature_set_table", "data"),
        )
        def update_all_plugin_properties(selected_rows, table_data):
            print("Updating Plugin Properties")
            print(f"Selected Rows: {selected_rows}")
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                raise PreventUpdate

            # Get the selected row data and grab the uuid
            selected_row_data = table_data[selected_rows[0]]
            object_uuid = selected_row_data["uuid"]

            # Create the FeatureSet object
            feature_set = FeatureSet(object_uuid)

            # Update all the properties for each plugin
            all_props = []
            for p in self.plugins:
                all_props.extend(p.update_properties(feature_set))

            # Return all the updated properties
            return all_props


# Unit Test for your Plugin Page
if __name__ == "__main__":
    # Note: This 'main' is purely for running/testing locally
    import webbrowser

    # Create our Dash Application
    my_app = dash.Dash(
        __name__,
        title="SageWorks Dashboard",
        use_pages=True,
        pages_folder="",
        external_stylesheets=[dbc.themes.DARKLY],
    )

    # For Multi-Page Applications, we need to create a 'page container' to hold all the pages
    my_app.layout = html.Div([page_container])

    # Create the Plugin Page and call page_setup
    plugin_page = MDQPluginPage()
    plugin_page.page_setup(my_app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/mdq_view")
    my_app.run(host="localhost", port=8000, debug=True)
