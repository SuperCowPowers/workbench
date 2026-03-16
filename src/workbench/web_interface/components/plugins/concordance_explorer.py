"""Concordance Explorer: Composite plugin with concordance map + neighbors table."""

import logging
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output
from dash.exceptions import PreventUpdate

from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.concordance_map import ConcordanceMap
from workbench.web_interface.components.plugins.ag_table import AGTable

log = logging.getLogger("workbench")


class ConcordanceExplorer(PluginInterface):
    """Composite plugin: ConcordanceMap (scatter) + AGTable (neighbor details on click).

    Clicking a compound on the map populates the table with its nearest neighbors.
    Clicking a table row highlights the corresponding scatter point (white circle +
    molecule tooltip) via a shared dcc.Store and a clientside synthetic mousemove.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def __init__(self):
        """Initialize the ConcordanceExplorer plugin."""
        self.component_id = None
        self.table = AGTable()
        self.dc = None
        self.concordance_map = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the composite component (map left, table right).

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div containing the concordance map and neighbors table.
        """
        self.component_id = component_id
        store_id = f"{component_id}-hover-store"

        self.concordance_map = ConcordanceMap(graph_height="600px", external_hover_id=store_id)
        map_component = self.concordance_map.create_component(f"{component_id}-map")
        table_component = self.table.create_component(f"{component_id}-table")

        self.properties = list(self.concordance_map.properties) + list(self.table.properties)
        self.signals = list(self.concordance_map.signals) + list(self.table.signals)

        return html.Div(
            id=component_id,
            children=[
                dcc.Store(id=store_id),
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
        self.dc = kwargs.pop("dc", None)

        map_props = self.concordance_map.update_properties(input_data, **kwargs)

        empty_df = pd.DataFrame(columns=["Click a compound to see neighbors"])
        table_props = self.table.update_properties(empty_df)

        return map_props + table_props

    def register_internal_callbacks(self):
        """Register cross-component callbacks for bidirectional interaction."""
        self.concordance_map.register_internal_callbacks()

        graph_id = f"{self.component_id}-map-graph"
        table_id = f"{self.component_id}-table"
        store_id = f"{self.component_id}-hover-store"

        # --- Map click → populate table with neighbors ---
        table_outputs = [Output(cid, prop, allow_duplicate=True) for cid, prop in self.table.properties]

        @callback(
            *table_outputs,
            Input(graph_id, "clickData"),
            prevent_initial_call=True,
        )
        def _update_table_on_click(click_data):
            if not click_data or "points" not in click_data or self.dc is None:
                raise PreventUpdate

            point = click_data["points"][0]
            customdata = point.get("customdata")
            if customdata is None or len(customdata) < 2:
                raise PreventUpdate

            compound_id = customdata[1]
            log.info(f"ConcordanceExplorer: clicked compound '{compound_id}'")

            neighbors_df = self.dc.neighbors(compound_id, n_neighbors=20)

            id_col = self.dc.id_column
            if id_col in neighbors_df.columns:
                neighbors_df = neighbors_df.drop(columns=[id_col])

            result = self.table.update_properties(neighbors_df)
            for col_def in result[0]:
                col_def["width"] = 150
            return result

        # --- Table row selection → write mol_id to Store ---
        # The Store triggers a clientside callback in ConcordanceMap that dispatches
        # a synthetic mousemove to trigger Plotly's real hover on the matching point.
        @callback(
            Output(store_id, "data"),
            Input(table_id, "selectedRows"),
            prevent_initial_call=True,
        )
        def _table_to_hover_store(selected_rows):
            if not selected_rows:
                raise PreventUpdate

            neighbor_id = selected_rows[0].get("neighbor_id")
            if not neighbor_id:
                raise PreventUpdate

            return {"mol_id": str(neighbor_id)}


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
