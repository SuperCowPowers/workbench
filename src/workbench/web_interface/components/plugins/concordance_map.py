"""Concordance Map: Standalone multi-layer visualization for dataset concordance analysis."""

import base64
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, callback, clientside_callback, Input, Output, no_update
from dash.exceptions import PreventUpdate

from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.clientside_callbacks import circle_overlay_callback
from workbench.utils.chem_utils.vis import molecule_hover_tooltip


class ConcordanceMap(PluginInterface):
    """Concordance visualization with concordance-aware coloring.

    Renders three layers (bottom to top):
        1. Reference compounds (dark grey)
        2. Novel query compounds with low chemical space overlap (blue)
        3. Overlapping query compounds colored by |target_residual| (green -> red)

    Uses go.Scatter (SVG) instead of Scattergl to enable Plotly.Fx.hover() for
    programmatic hover triggering from external components (e.g. table selection).

    Expects a unified DataFrame from DatasetConcordance.concordance_results()
    with columns: x, y, dataset, tanimoto_sim, target_residual, etc.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    # Colorscale for SAR concordance: green (concordant) -> red (discordant)
    CONCORDANCE_COLORSCALE = [
        [0.0, "rgb(54, 139, 54)"],  # forest green — concordant
        [0.25, "rgb(154, 205, 50)"],  # yellow-green
        [0.5, "rgb(225, 215, 20)"],  # gold
        [0.75, "rgb(225, 99, 71)"],  # tomato
        [1.0, "rgb(220, 60, 60)"],  # crimson — discordant
    ]

    # White circle overlay SVG for hover highlighting
    _circle_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" style="overflow: visible;">
        <circle cx="50" cy="50" r="10" stroke="rgba(255, 255, 255, 1)" stroke-width="3" fill="none" />
    </svg>"""
    _circle_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(_circle_svg.encode('utf-8')).decode('utf-8')}"

    def __init__(self, novel_threshold: float = 0.3, graph_height: str = "1200px"):
        """Initialize the ConcordanceMap plugin.

        Args:
            novel_threshold (float): Tanimoto similarity threshold below which query
                compounds are considered "novel" (default: 0.3).
            graph_height (str): CSS height for the graph component (default: "1200px").
        """
        self.novel_threshold = novel_threshold
        self.graph_height = graph_height
        self.concordance_cmax = None
        self.component_id = None
        self.df = None
        self.smiles_column = None
        self.id_column = None
        self.has_smiles = False
        self.hover_background = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the concordance map component.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div containing the graph and hover tooltips.
        """
        self.component_id = component_id

        self.properties = [(f"{component_id}-graph", "figure")]
        self.signals = [
            (f"{component_id}-graph", "hoverData"),
            (f"{component_id}-graph", "clickData"),
        ]

        return html.Div(
            children=[
                dcc.Graph(
                    id=f"{component_id}-graph",
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                    style={"height": self.graph_height, "width": "100%"},
                    clear_on_unhover=True,
                ),
                dcc.Tooltip(
                    id=f"{component_id}-overlay",
                    background_color="rgba(0,0,0,0)",
                    border_color="rgba(0,0,0,0)",
                    direction="bottom",
                    loading_text="",
                ),
                dcc.Tooltip(
                    id=f"{component_id}-molecule-tooltip",
                    background_color="rgba(0,0,0,0)",
                    border_color="rgba(0,0,0,0)",
                    direction="bottom",
                    loading_text="",
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Update the concordance map with new data.

        Args:
            input_data (pd.DataFrame): Unified DataFrame from DatasetConcordance.concordance_results().
            **kwargs (dict): Must include ``target_column`` and ``id_column``.

        Returns:
            list: Single-element list containing the Plotly figure.
        """
        self.hover_background = self.theme_manager.background()
        self.df = input_data

        # Detect SMILES and ID columns
        self.smiles_column = next((col for col in self.df.columns if col.lower() == "smiles"), None)
        self.id_column = kwargs.get("id_column") or next(
            (col for col in self.df.columns if col.lower() == "id"), self.df.columns[0]
        )
        self.has_smiles = self.smiles_column is not None

        # Compute concordance colorscale max from reference target variability
        target_column = kwargs.get("target_column")
        if target_column and target_column in input_data.columns and "dataset" in input_data.columns:
            ref_std = input_data.loc[input_data["dataset"] == "reference", target_column].std()
            self.concordance_cmax = 2.0 * ref_std
        else:
            self.concordance_cmax = None

        figure = self.create_scatter_plot(self.df, "x", "y")
        return [figure]

    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        """Build the concordance-specific multi-layer figure.

        Args:
            df (pd.DataFrame): The unified concordance DataFrame.
            x_col (str): Column for x-axis (typically "x").
            y_col (str): Column for y-axis (typically "y").

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
                go.Scatter(
                    x=ref_df[x_col],
                    y=ref_df[y_col],
                    mode="markers",
                    name="Reference Compounds",
                    showlegend=True,
                    hoverinfo="none",
                    customdata=ref_df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=15,
                        color="rgba(128, 128, 128, 0.75)",
                        line=dict(color="rgba(0, 0, 0, 0.25)", width=0.5),
                    ),
                )
            )

        # --- Layer 2: Novel query compounds (blue) ---
        if len(novel_df) > 0:
            figure.add_trace(
                go.Scatter(
                    x=novel_df[x_col],
                    y=novel_df[y_col],
                    mode="markers",
                    name=f"Novel Compounds (sim < {self.novel_threshold})",
                    showlegend=True,
                    hoverinfo="none",
                    customdata=novel_df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=15,
                        color="rgba(110, 110, 200, 0.75)",
                        line=dict(color="rgba(0, 0, 0, 0.25)", width=1),
                    ),
                )
            )

        # --- Layer 3: Overlapping query compounds (green -> red by |residual|) ---
        if len(overlap_df) > 0:
            abs_residual = overlap_df["target_residual"].abs()

            # Sort so high-residual (discordant) points render on top
            sort_idx = abs_residual.argsort()
            overlap_df = overlap_df.iloc[sort_idx]
            abs_residual = abs_residual.iloc[sort_idx]

            cmax = self.concordance_cmax if self.concordance_cmax else abs_residual.quantile(0.95)

            figure.add_trace(
                go.Scatter(
                    x=overlap_df[x_col],
                    y=overlap_df[y_col],
                    mode="markers",
                    name="",
                    legendgroup="overlap",
                    showlegend=False,
                    hoverinfo="none",
                    customdata=overlap_df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=15,
                        color=abs_residual,
                        colorscale=self.CONCORDANCE_COLORSCALE,
                        cmin=0,
                        cmax=cmax,
                        colorbar=dict(
                            title=dict(text="Median Residual", font=dict(size=10)),
                            thickness=8,
                            len=0.3,
                            x=0.98,
                            xanchor="right",
                            y=0.98,
                            yanchor="top",
                            xpad=0,
                            ypad=0,
                            tickfont=dict(size=10),
                        ),
                        opacity=0.9,
                        line=dict(color="rgba(0, 0, 0, 0.5)", width=1),
                    ),
                )
            )

            # Gradient legend: group title + color swatches
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
                figure.add_trace(go.Scatter(**trace_kwargs))

        # --- Layout ---
        figure.update_layout(
            margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0},
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(0, 0, 0, 0.3)",
                font=dict(size=12),
                groupclick="togglegroup",
            ),
            dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},
            uirevision="constant",
        )

        return figure

    def register_internal_callbacks(self):
        """Register hover callbacks for circle overlay and molecule tooltip."""

        # Clientside callback for circle overlay (runs in browser, no server round trip)
        clientside_callback(
            circle_overlay_callback(self._circle_data_uri),
            Output(f"{self.component_id}-overlay", "show"),
            Output(f"{self.component_id}-overlay", "bbox"),
            Output(f"{self.component_id}-overlay", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )

        # Server-side callback for molecule tooltip
        @callback(
            Output(f"{self.component_id}-molecule-tooltip", "show"),
            Output(f"{self.component_id}-molecule-tooltip", "bbox"),
            Output(f"{self.component_id}-molecule-tooltip", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )
        def _molecule_tooltip(hover_data):
            """Show molecule tooltip when hovering over a compound."""
            if hover_data is None or not self.has_smiles:
                return False, no_update, no_update

            customdata = hover_data["points"][0].get("customdata")
            if customdata is None:
                return False, no_update, no_update

            if isinstance(customdata, (list, tuple)):
                smiles = customdata[0]
                mol_id = customdata[1] if len(customdata) > 1 and self.id_column else None
            else:
                smiles = customdata
                mol_id = None

            mol_width, mol_height = 300, 200
            children = molecule_hover_tooltip(
                smiles, mol_id=mol_id, width=mol_width, height=mol_height, background=self.hover_background
            )

            bbox = hover_data["points"][0]["bbox"]
            center_x = (bbox["x0"] + bbox["x1"]) / 2
            center_y = (bbox["y0"] + bbox["y1"]) / 2
            adjusted_bbox = {
                "x0": center_x + 5,
                "x1": center_x + 5 + mol_width,
                "y0": center_y - mol_height - (mol_height + 50),
                "y1": center_y - (mol_height + 50),
            }
            return True, adjusted_bbox, children


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
