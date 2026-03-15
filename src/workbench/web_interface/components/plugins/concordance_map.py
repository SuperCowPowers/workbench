"""Concordance Map: Multi-layer visualization for dataset concordance analysis."""

import pandas as pd
import plotly.graph_objects as go
from dash import html

from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot


class ConcordanceMap(ScatterPlot):
    """Concordance visualization with concordance-aware coloring.

    Renders three layers (bottom to top):
        1. Reference compounds (dark grey)
        2. Novel query compounds with low chemical space overlap (blue)
        3. Overlapping query compounds colored by |target_residual| (green → red)

    Inherits ScatterPlot for Dash component infrastructure (graph, dropdowns,
    tooltips, callbacks) but overrides create_scatter_plot() for concordance-specific
    multi-layer encoding. The color dropdown is not used — concordance encoding
    is always applied.

    Expects a unified DataFrame from DatasetConcordance.concordance_results()
    with columns: x, y, dataset, tanimoto_sim, target_residual, etc.
    """

    # Colorscale for SAR concordance: green (concordant) → red (discordant)
    CONCORDANCE_COLORSCALE = [
        [0.0, "rgb(34, 139, 34)"],  # forest green — concordant
        [0.25, "rgb(154, 205, 50)"],  # yellow-green
        [0.5, "rgb(255, 215, 0)"],  # gold
        [0.75, "rgb(255, 99, 71)"],  # tomato
        [1.0, "rgb(220, 20, 60)"],  # crimson — discordant
    ]

    def __init__(self, novel_threshold: float = 0.3):
        """Initialize the ConcordanceMap plugin.

        Args:
            novel_threshold (float): Tanimoto similarity threshold below which query
                compounds are considered "novel" (outside the model's applicability domain).
                Default: 0.3 (standard ECFP4 dissimilarity boundary).
        """
        super().__init__(show_axes=False)
        self.novel_threshold = novel_threshold
        self.concordance_cmax = None  # Set in update_properties from reference target std

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

        # Compute concordance colorscale max from reference target variability
        # Residuals beyond 2 std are considered truly discordant (saturate at red)
        target_column = kwargs.get("target_column")
        if target_column and target_column in input_data.columns and "dataset" in input_data.columns:
            ref_std = input_data.loc[input_data["dataset"] == "reference", target_column].std()
            self.concordance_cmax = 2.0 * ref_std
        else:
            self.concordance_cmax = None  # Fall back to auto-scaling

        return super().update_properties(input_data, **kwargs)

    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        regression_line: bool = False,
    ) -> go.Figure:
        """Build the concordance-specific multi-layer figure.

        Args:
            df (pd.DataFrame): The unified concordance DataFrame.
            x_col (str): Column for x-axis (typically "x").
            y_col (str): Column for y-axis (typically "y").
            color_col (str): Ignored — concordance encoding is always used.
            regression_line (bool): Ignored for concordance maps.

        Returns:
            go.Figure: A Plotly Figure with reference, novel, and overlap layers.
        """
        figure = go.Figure()

        # Split data by dataset and overlap
        ref_df = df[df["dataset"] == "reference"]
        query_df = df[df["dataset"] == "query"]
        novel_mask = query_df["tanimoto_sim"] < self.novel_threshold
        novel_df = query_df[novel_mask]
        overlap_df = query_df[~novel_mask]

        # Build customdata for molecule hover (smiles at index 0, id at index 1)
        custom_data_cols = []
        if self.smiles_column and self.smiles_column in df.columns:
            custom_data_cols.append(self.smiles_column)
        if self.id_column and self.id_column in df.columns:
            if self.id_column not in custom_data_cols:
                custom_data_cols.insert(1 if custom_data_cols else 0, self.id_column)

        # --- Layer 1: Reference scatter (dark grey) ---
        if len(ref_df) > 0:
            figure.add_trace(
                go.Scattergl(
                    x=ref_df[x_col],
                    y=ref_df[y_col],
                    mode="markers",
                    name="Reference Compounds",
                    showlegend=True,
                    hoverinfo="none",
                    customdata=ref_df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=15,
                        color="rgba(128, 128, 128, 0.5)",
                        line=dict(color="rgba(0, 0, 0, 0.25)", width=0.5),
                    ),
                )
            )

        # --- Layer 3: Novel query compounds (blue) ---
        if len(novel_df) > 0:
            figure.add_trace(
                go.Scattergl(
                    x=novel_df[x_col],
                    y=novel_df[y_col],
                    mode="markers",
                    name=f"Novel Compounds (sim < {self.novel_threshold})",
                    showlegend=True,
                    hoverinfo="none",
                    customdata=novel_df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=15,
                        color="rgba(110, 110, 180, 0.5)",
                        line=dict(color="rgba(0, 0, 0, 0.25)", width=1),
                    ),
                )
            )

        # --- Layer 4: Overlapping query compounds (green → red by |residual|) ---
        if len(overlap_df) > 0:
            abs_residual = overlap_df["target_residual"].abs()

            # Sort so high-residual (discordant) points render on top
            sort_idx = abs_residual.argsort()
            overlap_df = overlap_df.iloc[sort_idx]
            abs_residual = abs_residual.iloc[sort_idx]

            # Set colorscale range: use reference target std so "good" concordance is all green
            cmax = self.concordance_cmax if self.concordance_cmax else abs_residual.quantile(0.95)
            marker_kwargs = dict(
                size=15,
                color=abs_residual,
                colorscale=self.CONCORDANCE_COLORSCALE,
                cmin=0,
                cmax=cmax,
                colorbar=dict(title="|target_residual|", thickness=10),
                opacity=0.9,
                line=dict(color="rgba(0, 0, 0, 0.5)", width=1),
            )

            figure.add_trace(
                go.Scattergl(
                    x=overlap_df[x_col],
                    y=overlap_df[y_col],
                    mode="markers",
                    name="",
                    legendgroup="overlap",
                    showlegend=False,
                    hoverinfo="none",
                    customdata=overlap_df[custom_data_cols] if custom_data_cols else None,
                    marker=marker_kwargs,
                )
            )

            # Gradient legend: group title + indented color swatches
            gradient_legend = [
                ("rgb(34, 139, 34)", "Low Target Difference"),
                ("rgb(255, 215, 0)", "Medium Target Difference"),
                ("rgb(220, 20, 60)", "High Target Difference"),
            ]
            for i, (color, name) in enumerate(gradient_legend):
                trace_kwargs = dict(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=f"<span style='font-size: 10px'>  {name}</span>",
                    legendgroup="overlap",
                    showlegend=True,
                )
                if i == 0:
                    trace_kwargs["legendgrouptitle"] = dict(
                        text=f"    Overlap Compounds (sim >= {self.novel_threshold})",
                        font=dict(size=12),
                    )
                figure.add_trace(go.Scattergl(**trace_kwargs))

        # --- Layout ---
        figure.update_layout(
            margin={"t": 20, "b": 55, "r": 0, "l": 35, "pad": 0},
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0, 0, 0, 0.3)",
                font=dict(size=12),
                groupclick="togglegroup",
            ),
            dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},
            uirevision="constant",
        )

        return figure


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
        ConcordanceMap,
        input_data=results_df,
        theme="dark",
        height="1000px",
        id_column=dc.id_column,
        target_column=dc.target_column,
    ).run()
