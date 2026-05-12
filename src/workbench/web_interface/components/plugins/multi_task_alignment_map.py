"""Multi-Task Alignment Map: chemical-space + per-aux alignment visualization.

Companion UI to ``MultiTaskAlignment``. Renders a UMAP scatter with primary compounds
as the background and an aux-selector that swaps the foreground:

    - Primary background: dark grey, low alpha — the chemical-space "canvas"
    - Aux-only compounds (extension region): blue
    - Shared compounds (overlap region): colored by ``|residual_<aux>|``
      (green = aligned, red = discordant)

Clicking a compound populates a side table with its nearest neighbors. Selecting
a row in the table highlights the corresponding scatter point.
"""

import logging

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, clientside_callback, dcc, html, no_update
from dash.exceptions import PreventUpdate

from workbench.utils.chem_utils.vis import molecule_hover_tooltip
from workbench.web_interface.components.plugin_interface import (
    PluginInputType,
    PluginInterface,
    PluginPage,
)
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.web_interface.utils.clientside_callbacks import (
    external_highlight_callback,
    hover_ring_overlay_callback,
)

_HOVER_OVERLAY_NAME = "__hover_overlay__"

log = logging.getLogger("workbench")


class MultiTaskAlignmentMap(PluginInterface):
    """Chemical-space + per-aux alignment map for a ``MultiTaskAlignment``.

    Layout: scatter map (left) + neighbors table (right). The map has an aux-selector
    dropdown; switching auxes re-colors the foreground without rebuilding UMAP.

    Expects a ``MultiTaskAlignment`` instance passed via ``mta=`` to ``update_properties``.
    The unified DataFrame is taken from ``mta.results()``.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    # Green (aligned) -> red (discordant) for |residual_<aux>| coloring
    ALIGNMENT_COLORSCALE = [
        [0.0, "rgb(54, 139, 54)"],
        [0.25, "rgb(154, 205, 50)"],
        [0.5, "rgb(225, 215, 20)"],
        [0.75, "rgb(225, 99, 71)"],
        [1.0, "rgb(220, 60, 60)"],
    ]

    def __init__(
        self,
        novel_threshold: float = 0.3,
        residual_cmax: float = 2.0,
        graph_height: str = "600px",
    ) -> None:
        """Initialize the plugin.

        Args:
            novel_threshold: Tanimoto-to-primary threshold below which an aux compound is
                drawn in the "extension" (blue) layer instead of the overlap colored layer.
            residual_cmax: Upper end of the |residual| colorscale, in z-score units (residuals
                in ``MultiTaskAlignment.results()`` are already z-scored). Default 2.0 std units.
            graph_height: CSS height for the scatter graph.
        """
        self.novel_threshold = novel_threshold
        self.residual_cmax = residual_cmax
        self.graph_height = graph_height
        self.mta = None
        self.df = None
        self.primary = None
        self.auxiliaries: list[str] = []
        self.id_column = "id"
        self.smiles_column: str = "smiles"
        self.component_id = None
        self.hover_background = None
        super().__init__()

    # ------------------------------------------------------------------
    # Component layout
    # ------------------------------------------------------------------

    def create_component(self, component_id: str) -> html.Div:
        self.component_id = component_id
        graph_id = f"{component_id}-graph"
        molecule_tt_id = f"{component_id}-molecule-tooltip"
        table_id = f"{component_id}-table"
        store_id = f"{component_id}-hover-store"
        aux_dd_id = f"{component_id}-aux-dropdown"

        self.table = AGTable()
        table_component = self.table.create_component(table_id)

        self.properties = [
            (graph_id, "figure"),
            (aux_dd_id, "options"),
            (aux_dd_id, "value"),
        ] + list(self.table.properties)
        self.signals = [
            (graph_id, "hoverData"),
            (graph_id, "clickData"),
            (aux_dd_id, "value"),
        ] + list(self.table.signals)

        graph_panel = html.Div(
            children=[
                html.Div(
                    [
                        html.Label("Auxiliary:", style={"marginRight": "8px"}),
                        dcc.Dropdown(
                            id=aux_dd_id,
                            options=[],
                            value=None,
                            clearable=False,
                            style={"width": "240px", "display": "inline-block"},
                        ),
                    ],
                    style={"padding": "5px 5px 0 5px"},
                ),
                dcc.Graph(
                    id=graph_id,
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                    style={"height": self.graph_height, "width": "100%"},
                    clear_on_unhover=True,
                ),
                dcc.Tooltip(
                    id=molecule_tt_id,
                    background_color="rgba(0,0,0,0)",
                    border_color="rgba(0,0,0,0)",
                    direction="bottom",
                    loading_text="",
                ),
                # Dummy output target for the hover-ring clientside callback
                # (the JS does an imperative Plotly.restyle and returns no_update,
                # so this Store is never actually written).
                dcc.Store(id=f"{component_id}-hover-circle-dummy-output"),
            ],
            style={"display": "flex", "flexDirection": "column"},
        )

        return html.Div(
            id=component_id,
            children=[
                dcc.Store(id=store_id),
                dbc.Row(
                    [
                        dbc.Col(html.Div(graph_panel, className="workbench-container"), width=7),
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
                    ]
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Property updates
    # ------------------------------------------------------------------

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Update map + dropdown + (empty) table from a ``MultiTaskAlignment``.

        Args:
            input_data: ``MultiTaskAlignment.results()`` output (per-compound DataFrame
                with ``x``, ``y``, ``tanimoto_to_primary``, and ``residual_<aux>`` columns).
            **kwargs: Must include ``mta`` (the ``MultiTaskAlignment`` instance, used for
                neighbor lookups on click). Optionally ``aux=<name>`` to set the initial aux.

        Returns:
            list: figure + dropdown options + dropdown value + table props.
        """
        self.hover_background = self.theme_manager.background()
        self.mta = kwargs.get("mta")
        if self.mta is None:
            raise ValueError("MultiTaskAlignmentMap requires the `mta` kwarg (MultiTaskAlignment instance)")

        self.df = input_data
        self.primary = self.mta.primary
        self.auxiliaries = list(self.mta.auxiliaries)
        self.id_column = self.mta.id_column
        self.smiles_column = next((c for c in input_data.columns if c.lower() == "smiles"), "smiles")

        initial_aux = kwargs.get("aux") or (self.auxiliaries[0] if self.auxiliaries else None)
        figure = self._build_figure(initial_aux)

        options = [{"label": a, "value": a} for a in self.auxiliaries]
        empty_table = pd.DataFrame(columns=["Click a compound to see neighbors"])
        table_props = self.table.update_properties(empty_table)

        return [figure, options, initial_aux, *table_props]

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self, aux: str) -> go.Figure:
        # NOTE: traces below use go.Scatter (SVG), not go.Scattergl (WebGL).
        # The table-row-click highlight (see external_highlight_callback) needs
        # per-point DOM <path> elements to compute pixel position; WebGL points
        # are canvas pixels with no DOM handle. Until we have a renderer-agnostic
        # way to trigger Plotly hover by trace+point index, this plugin stays SVG.
        df = self.df
        figure = go.Figure()
        custom_cols = [c for c in (self.smiles_column, self.id_column) if c in df.columns]

        primary_mask = df[self.primary].notna() if self.primary in df.columns else pd.Series(False, index=df.index)
        if aux is None or aux not in df.columns:
            self._add_layer(figure, df, custom_cols, name="All Compounds", color="rgba(128, 128, 128, 0.75)")
            self._add_hover_overlay(figure)
            self._apply_layout(figure)
            return figure

        aux_mask = df[aux].notna()
        primary_only = df[primary_mask & ~aux_mask]
        aux_only = df[aux_mask & ~primary_mask]
        shared = df[primary_mask & aux_mask]

        # Aux-only compounds split on chemical-space coverage:
        #   tanimoto_to_primary < threshold -> "novel extension" (blue)
        #   tanimoto_to_primary >= threshold + has residual -> color by |residual|
        cov = aux_only["tanimoto_to_primary"] if "tanimoto_to_primary" in aux_only.columns else None
        if cov is not None:
            novel = aux_only[cov < self.novel_threshold]
            extension_overlap = aux_only[cov >= self.novel_threshold]
        else:
            novel = aux_only.iloc[0:0]
            extension_overlap = aux_only

        # Layer 1: primary-only compounds (canvas)
        self._add_layer(
            figure,
            primary_only,
            custom_cols,
            name=f"Primary only ({self.primary})",
            color="rgba(128, 128, 128, 0.55)",
        )

        # Layer 2: aux-only novel compounds (blue)
        self._add_layer(
            figure,
            novel,
            custom_cols,
            name=f"Novel {aux} (sim < {self.novel_threshold})",
            color="rgba(110, 110, 200, 0.85)",
        )

        # Layer 3: aux compounds with primary coverage — colored by residual
        residual_col = f"residual_{aux}"
        colored = pd.concat([shared, extension_overlap], ignore_index=False)
        if residual_col in colored.columns and len(colored) > 0:
            abs_res = colored[residual_col].abs()
            # Sort so high-residual points render on top
            order = abs_res.sort_values().index
            colored = colored.loc[order]
            abs_res = abs_res.loc[order]
            figure.add_trace(
                go.Scatter(
                    x=colored["x"],
                    y=colored["y"],
                    mode="markers",
                    name=f"Aligned/Discordant ({aux})",
                    legendgroup="overlap",
                    showlegend=False,
                    hoverinfo="none",
                    customdata=colored[custom_cols] if custom_cols else None,
                    marker=dict(
                        size=12,
                        color=abs_res,
                        colorscale=self.ALIGNMENT_COLORSCALE,
                        cmin=0,
                        cmax=self.residual_cmax,
                        colorbar=dict(
                            title=dict(text=f"|residual {aux}|", font=dict(size=10)),
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
            for i, (color, label) in enumerate(
                [
                    ("rgb(34, 139, 34)", "Aligned"),
                    ("rgb(255, 215, 0)", "Partial"),
                    ("rgb(220, 20, 60)", "Discordant"),
                ]
            ):
                trace_kwargs = dict(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=f"<span style='font-size: 10px'>  {label}</span>",
                    legendgroup="overlap",
                    showlegend=True,
                )
                if i == 0:
                    trace_kwargs["legendgrouptitle"] = dict(
                        text=f"    {aux} alignment (sim >= {self.novel_threshold})",
                        font=dict(size=12),
                    )
                figure.add_trace(go.Scatter(**trace_kwargs))

        self._add_hover_overlay(figure)
        self._apply_layout(figure)
        return figure

    def _add_layer(
        self,
        figure: go.Figure,
        df: pd.DataFrame,
        custom_cols: list,
        name: str,
        color: str,
    ) -> None:
        if len(df) == 0:
            return
        figure.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                name=name,
                showlegend=True,
                hoverinfo="none",
                customdata=df[custom_cols] if custom_cols else None,
                marker=dict(size=12, color=color, line=dict(color="rgba(0, 0, 0, 0.25)", width=0.5)),
            )
        )

    @staticmethod
    def _add_hover_overlay(figure: go.Figure) -> None:
        """Append a hidden single-point ring trace for the clientside hover callback."""
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                hoverinfo="skip",
                showlegend=False,
                name=_HOVER_OVERLAY_NAME,
                marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="white", width=3)),
            )
        )

    @staticmethod
    def _apply_layout(figure: go.Figure) -> None:
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

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_internal_callbacks(self):
        graph_id = f"{self.component_id}-graph"
        molecule_tt_id = f"{self.component_id}-molecule-tooltip"
        table_id = f"{self.component_id}-table"
        store_id = f"{self.component_id}-hover-store"
        aux_dd_id = f"{self.component_id}-aux-dropdown"

        # Aux-selector → rebuild figure for the selected aux
        @callback(
            Output(graph_id, "figure", allow_duplicate=True),
            Input(aux_dd_id, "value"),
            prevent_initial_call=True,
        )
        def _on_aux_change(aux):
            if aux is None or self.df is None:
                raise PreventUpdate
            return self._build_figure(aux)

        # External highlight via shared store (table row → synthetic mousemove on the point)
        clientside_callback(
            external_highlight_callback(graph_id),
            Output(graph_id, "id"),
            Input(store_id, "data"),
            prevent_initial_call=True,
        )

        # Hover-ring overlay: JS does an imperative Plotly.restyle and returns no_update.
        # The Store Output is a placeholder that's never actually written.
        clientside_callback(
            hover_ring_overlay_callback(graph_id, _HOVER_OVERLAY_NAME),
            Output(f"{self.component_id}-hover-circle-dummy-output", "data"),
            Input(graph_id, "hoverData"),
        )

        # Molecule tooltip on hover
        @callback(
            Output(molecule_tt_id, "show"),
            Output(molecule_tt_id, "bbox"),
            Output(molecule_tt_id, "children"),
            Input(graph_id, "hoverData"),
        )
        def _molecule_tooltip(hover_data):
            if not hover_data:
                return False, no_update, no_update
            customdata = hover_data["points"][0].get("customdata")
            if customdata is None:
                return False, no_update, no_update
            if isinstance(customdata, (list, tuple)):
                smiles = customdata[0]
                mol_id = customdata[1] if len(customdata) > 1 else None
            else:
                smiles = customdata
                mol_id = None

            mol_w, mol_h = 300, 200
            children = molecule_hover_tooltip(
                smiles, mol_id=mol_id, width=mol_w, height=mol_h, background=self.hover_background
            )
            bbox = hover_data["points"][0]["bbox"]
            cx = (bbox["x0"] + bbox["x1"]) / 2
            cy = (bbox["y0"] + bbox["y1"]) / 2
            adjusted_bbox = {
                "x0": cx + 5,
                "x1": cx + 5 + mol_w,
                "y0": cy - mol_h - (mol_h + 50),
                "y1": cy - (mol_h + 50),
            }
            return True, adjusted_bbox, children

        # Click on graph → populate neighbors table
        table_outputs = [Output(cid, prop, allow_duplicate=True) for cid, prop in self.table.properties]

        @callback(
            *table_outputs,
            Input(graph_id, "clickData"),
            prevent_initial_call=True,
        )
        def _on_click(click_data):
            if not click_data or "points" not in click_data or self.mta is None:
                raise PreventUpdate
            customdata = click_data["points"][0].get("customdata")
            if customdata is None or len(customdata) < 2:
                raise PreventUpdate
            compound_id = customdata[1]
            log.info(f"MultiTaskAlignmentMap: clicked compound '{compound_id}'")

            neighbors_df = self.mta.neighbors(compound_id, n_neighbors=20)
            if self.id_column in neighbors_df.columns:
                neighbors_df = neighbors_df.drop(columns=[self.id_column])
            result = self.table.update_properties(neighbors_df)
            for col_def in result[0]:
                col_def["width"] = 150
            return result

        # Table row → write mol_id to store (triggers external highlight clientside)
        @callback(
            Output(store_id, "data"),
            Input(table_id, "selectedRows"),
            prevent_initial_call=True,
        )
        def _row_to_store(selected_rows):
            if not selected_rows:
                raise PreventUpdate
            neighbor_id = selected_rows[0].get("neighbor_id")
            if not neighbor_id:
                raise PreventUpdate
            return {"mol_id": str(neighbor_id)}


if __name__ == "__main__":
    from workbench.algorithms.dataframe.multi_task_alignment import MultiTaskAlignment
    from workbench.utils.synthetic_data_generator import SyntheticDataGenerator
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Build a synthetic two-target MT df from AQSol partitions
    ref_df, query_df = SyntheticDataGenerator().aqsol_alignment_data(overlap="low", alignment="high")
    ref_df = ref_df.assign(id=ref_df["id"].astype(str).radd("ref_"))
    query_df = query_df.assign(id=query_df["id"].astype(str).radd("qry_"))
    ref_df = ref_df.rename(columns={"solubility": "primary_sol"})
    query_df = query_df.rename(columns={"solubility": "aux_sol"})
    mt_df = pd.concat(
        [
            ref_df[["id", "smiles", "primary_sol"]],
            query_df[["id", "smiles", "aux_sol"]],
        ],
        ignore_index=True,
    )

    mta = MultiTaskAlignment(mt_df, primary="primary_sol", auxiliaries=["aux_sol"], id_column="id")

    PluginUnitTest(
        MultiTaskAlignmentMap,
        input_data=mta.results(),
        theme="dark",
        mta=mta,
    ).run()
