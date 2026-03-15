"""Concordance Map: Thin wrapper around ScatterPlot for concordance visualization."""

import pandas as pd
from dash import html

from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot


class ConcordanceMap(ScatterPlot):
    """Scatter plot for dataset concordance, colored by SAR concordance residual.

    Expects a unified DataFrame from DatasetConcordance.concordance_results()
    with columns: x, y, dataset, tanimoto_sim, target_residual, etc.
    """

    def __init__(self):
        """Initialize the ConcordanceMap plugin."""
        super().__init__(show_axes=False)

    def create_component(self, component_id: str) -> html.Div:
        """Create the component (delegates to ScatterPlot).

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div Component containing the scatter plot.
        """
        component = super().create_component(component_id)

        # Make the graph larger for concordance visualization
        graph = component.children[0]
        graph.style["height"] = "1200px"

        return component

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Set concordance-specific defaults and delegate to ScatterPlot.

        Args:
            input_data (pd.DataFrame): Unified DataFrame from DatasetConcordance.concordance_results().
            **kwargs (dict): Additional keyword arguments (passed through to ScatterPlot).

        Returns:
            list: A list of updated property values from ScatterPlot.
        """
        kwargs.setdefault("x", "x")
        kwargs.setdefault("y", "y")
        kwargs.setdefault("color", "target_residual")
        return super().update_properties(input_data, **kwargs)


if __name__ == "__main__":
    from workbench.api import FeatureSet
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.utils.test_data_generator import TestDataGenerator
    from workbench.algorithms.dataframe.dataset_concordance import DatasetConcordance

    # Build concordance data
    ref_df, query_df = TestDataGenerator().aqsol_alignment_data(overlap="medium", alignment="high")
    # dc = DatasetConcordance(ref_df, query_df, target_column="solubility", id_column="id")

    # Temp test
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
        ConcordanceMap,
        input_data=results_df,
        theme="dark",
        id_column=dc.id_column,
        target_column=dc.target_column,
    ).run()
