"""Dataset Alignment Map: Thin wrapper around ScatterPlot for alignment visualization."""

import pandas as pd
from dash import html

from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot


class DatasetAlignmentMap(ScatterPlot):
    """Scatter plot for dataset alignment, colored by median_ref_residual.

    Expects a unified DataFrame from DatasetAlignment.dataset_alignment_results()
    with columns: x, y, dataset, highest_ref_tanimoto, median_ref_residual, etc.
    """

    def __init__(self):
        """Initialize the DatasetAlignmentMap plugin."""
        super().__init__(show_axes=False)

    def create_component(self, component_id: str) -> html.Div:
        """Create the component (delegates to ScatterPlot).

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div Component containing the scatter plot.
        """
        component = super().create_component(component_id)

        # Make the graph larger for alignment visualization
        graph = component.children[0]
        graph.style["height"] = "1200px"

        return component

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Set alignment-specific defaults and delegate to ScatterPlot.

        Args:
            input_data (pd.DataFrame): Unified DataFrame from DatasetAlignment.dataset_alignment_results().
            **kwargs (dict): Additional keyword arguments (passed through to ScatterPlot).

        Returns:
            list: A list of updated property values from ScatterPlot.
        """
        kwargs.setdefault("x", "x")
        kwargs.setdefault("y", "y")
        kwargs.setdefault("color", "median_ref_residual")
        return super().update_properties(input_data, **kwargs)


if __name__ == "__main__":
    from workbench.api import FeatureSet
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.utils.test_data_generator import TestDataGenerator
    from workbench.algorithms.dataframe.dataset_alignment import DatasetAlignment

    # Build alignment data
    # ref_df, query_df = TestDataGenerator().aqsol_alignment_data(overlap="medium", alignment="high")
    # da = DatasetAlignment(ref_df, query_df, target_column="solubility", id_column="id")

    # Temp test
    fs = FeatureSet("logd_value_reg_1")
    df = fs.pull_dataframe()
    id_col = "udm_mol_bat_id"
    is_doi = df["udm_asy_protocol"] == "DOI"
    ref_df = df[~is_doi]
    query_df = df[is_doi]
    target = "udm_asy_res_value"
    da = DatasetAlignment(ref_df, query_df, target_column=target, id_column=id_col)

    # Run the plugin unit test with the unified DataFrame
    PluginUnitTest(
        DatasetAlignmentMap,
        input_data=da.dataset_alignment_results(),
        theme="dark",
        id_column=da.id_column,
        target_column=da.target_column,
    ).run()
