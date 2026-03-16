"""Concordance Explorer: Composite plugin with concordance map + neighbors table."""

import logging
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, callback, clientside_callback, Input, Output
from dash.exceptions import PreventUpdate

from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.concordance_map import ConcordanceMap
from workbench.web_interface.components.plugins.ag_table import AGTable

log = logging.getLogger("workbench")


class ConcordanceExplorer(PluginInterface):
    """Composite plugin: ConcordanceMap (scatter) + AGTable (neighbor details on click).

    Clicking a compound on the map populates the table with its nearest neighbors
    from the DatasetConcordance proximity model. Clicking a table row highlights
    the selected compound on the scatter plot.

    Expects a unified DataFrame from DatasetConcordance.concordance_results()
    and a ``dc`` kwarg with the DatasetConcordance object for neighbor lookups.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def __init__(self):
        """Initialize the ConcordanceExplorer plugin."""
        self.component_id = None
        self.concordance_map = ConcordanceMap(graph_height="600px")
        self.table = AGTable()
        self.dc = None  # DatasetConcordance object for neighbor lookups
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the composite component (map left, table right).

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div containing the concordance map and neighbors table.
        """
        self.component_id = component_id

        # Create child components with namespaced IDs
        map_component = self.concordance_map.create_component(f"{component_id}-map")
        table_component = self.table.create_component(f"{component_id}-table")

        # Aggregate properties and signals from children
        self.properties = list(self.concordance_map.properties) + list(self.table.properties)
        self.signals = list(self.concordance_map.signals) + list(self.table.signals)

        return html.Div(
            id=component_id,
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(map_component, className="workbench-container"),
                            width=7,
                        ),
                        dbc.Col(
                            [
                                html.H4(
                                    "Compound Neighbors",
                                    style={"marginBottom": "5px", "paddingLeft": "5px"},
                                ),
                                table_component,
                            ],
                            width=5,
                            style={"paddingLeft": "0"},
                        ),
                    ],
                ),
            ],
        )

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Update properties for both child plugins.

        Args:
            input_data (pd.DataFrame): Unified DataFrame from DatasetConcordance.concordance_results().
            **kwargs (dict): Must include ``dc`` (DatasetConcordance object). Other kwargs
                are passed through to ConcordanceMap.

        Returns:
            list: A list of updated property values (map props + table props).
        """
        # Store the DatasetConcordance object for neighbor lookups on click
        self.dc = kwargs.pop("dc", None)

        # Update the concordance map
        map_props = self.concordance_map.update_properties(input_data, **kwargs)

        # Initialize the table with an empty DataFrame (no compound clicked yet)
        empty_df = pd.DataFrame(columns=["Click a compound to see neighbors"])
        table_props = self.table.update_properties(empty_df)

        return map_props + table_props

    def register_internal_callbacks(self):
        """Register cross-component callbacks for bidirectional interaction."""

        # Register the concordance map's own internal callbacks (hover tooltip, dropdowns)
        self.concordance_map.register_internal_callbacks()

        # --- Callback 1: Map click → Table update ---
        table_outputs = [Output(cid, prop, allow_duplicate=True) for cid, prop in self.table.properties]

        @callback(
            *table_outputs,
            Input(f"{self.component_id}-map-graph", "clickData"),
            prevent_initial_call=True,
        )
        def _update_table_on_click(click_data):
            """When a compound is clicked, populate the table with its neighbors."""
            if not click_data or "points" not in click_data or self.dc is None:
                raise PreventUpdate

            # Extract compound ID from customdata (index 1 = id_column)
            point = click_data["points"][0]
            customdata = point.get("customdata")
            if customdata is None or len(customdata) < 2:
                raise PreventUpdate

            compound_id = customdata[1]
            log.info(f"ConcordanceExplorer: clicked compound '{compound_id}'")

            # Get neighbors from the DatasetConcordance object
            neighbors_df = self.dc.neighbors(compound_id, n_neighbors=20)

            # Drop the first column (id_column) — it just repeats the clicked compound ID
            id_col = self.dc.id_column
            if id_col in neighbors_df.columns:
                neighbors_df = neighbors_df.drop(columns=[id_col])

            # Update the table and set compact column widths so more columns are visible
            result = self.table.update_properties(neighbors_df)
            column_defs = result[0]
            for col_def in column_defs:
                col_def["width"] = 120
            return result

        # --- Callback 2: Table row click → Fake hover on scatter plot ---
        # Clientside callback searches traces for the selected neighbor_id and triggers Plotly hover
        graph_id = f"{self.component_id}-map-graph"
        table_id = f"{self.component_id}-table"
        clientside_callback(
            """
            function(selectedRows) {
                if (!selectedRows || selectedRows.length === 0) {
                    return window.dash_clientside.no_update;
                }
                var neighborId = selectedRows[0].neighbor_id;
                if (!neighborId) {
                    return window.dash_clientside.no_update;
                }

                // Find the graph element and search its traces for the matching compound
                var graphEl = document.getElementById('"""
            + graph_id
            + """');
                if (!graphEl || !graphEl.data) {
                    return window.dash_clientside.no_update;
                }

                for (var ci = 0; ci < graphEl.data.length; ci++) {
                    var trace = graphEl.data[ci];
                    if (!trace.customdata) continue;
                    for (var pi = 0; pi < trace.customdata.length; pi++) {
                        var cd = trace.customdata[pi];
                        // customdata[1] is the compound ID
                        if (cd && cd.length > 1 && cd[1] === neighborId) {
                            Plotly.Fx.hover(graphEl, [{curveNumber: ci, pointNumber: pi}]);
                            return window.dash_clientside.no_update;
                        }
                    }
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output(table_id, "id"),  # Dummy output (no-op, just returns no_update)
            Input(table_id, "selectedRows"),
            prevent_initial_call=True,
        )


if __name__ == "__main__":
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.utils.test_data_generator import TestDataGenerator
    from workbench.algorithms.dataframe.dataset_concordance import DatasetConcordance

    # Unit test: synthetic test data
    ref_df, query_df = TestDataGenerator().aqsol_alignment_data(overlap="low", alignment="high")
    dc = DatasetConcordance(ref_df, query_df, target_column="solubility", id_column="id")
    id_col = "id"
    target = "solubility"

    # Temporary client test (comment out unit test above and uncomment below)
    from workbench.api import FeatureSet

    fs = FeatureSet("caco2_pappab_reg_1")
    df = fs.pull_dataframe()
    id_col = "udm_mol_bat_id"
    is_doi = df["udm_asy_protocol"] == "DOI"
    ref_df = df[~is_doi]
    query_df = df[is_doi]
    target = "udm_asy_res_pappa_b_10_6_cm_per_s"
    dc = DatasetConcordance(ref_df, query_df, target_column=target, id_column=id_col)

    # Only use the columns of interest for the plugin unit test
    results_df = dc.concordance_results()
    cols = [id_col, "smiles", "x", "y", "dataset", target, "tanimoto_sim", "target_residual"]
    results_df = results_df[[c for c in cols if c in results_df.columns]]

    # Run the plugin unit test with the unified DataFrame
    PluginUnitTest(
        ConcordanceExplorer,
        input_data=results_df,
        theme="dark",
        dc=dc,
        id_column=dc.id_column,
        target_column=dc.target_column,
    ).run()
