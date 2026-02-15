"""A Confusion Triangle (Ternary Plot) plugin for 3-class classification models."""

import base64
import numpy as np
import pandas as pd
from dash import dcc, html, callback, clientside_callback, Input, Output, no_update
import plotly.graph_objects as go
import plotly.express as pex
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.cached.cached_model import CachedModel
from workbench.utils.clientside_callbacks import circle_overlay_callback
from workbench.utils.chem_utils.vis import molecule_hover_tooltip
from workbench.utils.color_utils import add_alpha_to_first_color

# Marker style constants
_MARKER_LINE = dict(color="rgba(0,0,0,0.25)", width=1)
_DIMMED_MARKER = dict(size=10, color="rgba(128, 128, 128, 0.3)", line=dict(color="rgba(0,0,0,0.3)", width=1))


class ConfusionTriangle(PluginInterface):
    """A Confusion Triangle Plugin for 3-class classification models.

    The three class probabilities (which sum to 1) are plotted on a ternary simplex:
    - Bottom-left vertex: lowest class (first in class_labels)
    - Top vertex: middle class
    - Bottom-right vertex: highest class (last in class_labels)
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    # Pre-computed circle overlay SVG (same as scatter_plot)
    _circle_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" style="overflow: visible;">
        <circle cx="50" cy="50" r="10" stroke="rgba(255, 255, 255, 1)" stroke-width="3" fill="none" />
    </svg>"""
    _circle_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(_circle_svg.encode('utf-8')).decode('utf-8')}"

    def __init__(self):
        """Initialize the ConfusionTriangle plugin class."""
        self.component_id = None
        self.model = None
        self.inference_run = None
        self.df = None
        self.class_labels = None
        self.proba_cols = None
        self.target_col = None
        self.default_color = None  # Optional: set by parent to override default color column
        self.has_smiles = False
        self.smiles_column = None
        self.id_column = None
        self.hover_background = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Dash Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div Component containing the ternary graph and color dropdown.
        """
        self.component_id = component_id

        # Fill in plugin properties and signals
        self.properties = [
            (f"{component_id}-graph", "figure"),
            (f"{component_id}-color-dropdown", "options"),
            (f"{component_id}-color-dropdown", "value"),
        ]
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
                    style={"height": "400px", "width": "100%"},
                    clear_on_unhover=True,
                ),
                html.Div(
                    [
                        html.Div(style={"flex": 1}),
                        html.Label("Color", style={"marginRight": "5px", "fontWeight": "bold",
                                                    "display": "flex", "alignItems": "center"}),
                        dcc.Dropdown(id=f"{component_id}-color-dropdown", style={"minWidth": "200px"},
                                     clearable=False),
                    ],
                    style={"padding": "0px 20px 10px 20px", "display": "flex", "alignItems": "center", "gap": "5px"},
                ),
                dcc.Tooltip(id=f"{component_id}-overlay", background_color="rgba(0,0,0,0)",
                            border_color="rgba(0,0,0,0)", direction="bottom", loading_text=""),
                dcc.Tooltip(id=f"{component_id}-molecule-tooltip", background_color="rgba(0,0,0,0)",
                            border_color="rgba(0,0,0,0)", direction="bottom", loading_text=""),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            model (CachedModel): Workbench instantiated CachedModel object.
            **kwargs (dict):
                - inference_run (str): Inference capture name (default: "full_cross_fold").

        Returns:
            list: A list of updated property values [figure, color_options, color_default].
        """
        self.hover_background = self.theme_manager.background()
        self.model = model
        self.inference_run = kwargs.get("inference_run", "full_cross_fold")

        # Get class labels and validate we have exactly 3
        self.class_labels = model.class_labels()
        if self.class_labels is None or len(self.class_labels) != 3:
            n = len(self.class_labels) if self.class_labels else 0
            return [self.display_text(f"Requires 3-class classifier (got {n} classes)"), [], None]

        # Get inference predictions and drop rows with NaN in key columns
        self.df = model.get_inference_predictions(self.inference_run)
        if self.df is None or self.df.empty:
            return [self.display_text("No Prediction Data"), [], None]
        self.proba_cols = [f"{label}_proba" for label in self.class_labels]
        missing = [col for col in self.proba_cols if col not in self.df.columns]
        if missing:
            return [self.display_text(f"Missing columns: {missing}"), [], None]

        # Drop rows with NaN in probability or residual columns (bad predictions)
        drop_cols = self.proba_cols + (["residual"] if "residual" in self.df.columns else [])
        self.df = self.df.dropna(subset=drop_cols).reset_index(drop=True)

        # Detect smiles and id columns for molecule hover rendering
        self.smiles_column = next((col for col in self.df.columns if col.lower() == "smiles"), None)
        self.id_column = next((col for col in self.df.columns if col.lower() == "id"), self.df.columns[0])
        self.has_smiles = self.smiles_column is not None

        # Build color dropdown options
        self.target_col = model.target()
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        cat_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        cat_columns = [col for col in cat_columns if self.df[col].astype(str).nunique() < 20]
        color_columns = numeric_columns + cat_columns

        # Use default_color if set by parent (e.g., ConfusionExplorer sets "residual")
        if self.default_color and self.default_color in color_columns:
            color_default = self.default_color
        else:
            color_default = self.target_col if self.target_col in color_columns else (
                color_columns[0] if color_columns else None)
        color_options = [{"label": col, "value": col} for col in color_columns]

        figure = self.create_ternary_plot(self.df, self.class_labels, self.proba_cols, color_default)
        return [figure, color_options, color_default]

    def set_theme(self, theme: str) -> list:
        """Re-render the confusion triangle when the theme changes."""
        if self.model is None:
            return [no_update] * len(self.properties)
        return self.update_properties(self.model, inference_run=self.inference_run)

    @staticmethod
    def _project(low, mid, high):
        """Project barycentric coordinates to 2D cartesian (equilateral triangle).

        Args:
            low (float | np.ndarray): Probability for the lowest class.
            mid (float | np.ndarray): Probability for the middle class.
            high (float | np.ndarray): Probability for the highest class.

        Returns:
            tuple: (x, y) cartesian coordinates.
        """
        h = np.sqrt(3) / 2
        return high + mid * 0.5, mid * h

    def create_ternary_plot(self, df, class_labels, proba_cols, color_col, mask=None):
        """Create a ternary scatter plot using Scattergl with manual 2D projection.

        When mask is None, all points are rendered normally. When mask is provided,
        matching points (True) are highlighted and non-matching points (False) are
        dimmed to grey for selection-style linked brushing.

        Args:
            df (pd.DataFrame): The dataframe containing prediction data.
            class_labels (list): The three class labels [low, mid, high].
            proba_cols (list): The three probability column names.
            color_col (str): The column to use for coloring points.
            mask (pd.Series): Optional boolean mask for selection-style brushing.

        Returns:
            go.Figure: A Plotly Figure object.
        """
        h = np.sqrt(3) / 2
        px_arr, py_arr = self._project(df[proba_cols[0]].values, df[proba_cols[1]].values, df[proba_cols[2]].values)

        # Build customdata columns for molecule hover
        custom_data_cols = []
        if self.has_smiles:
            custom_data_cols.append(self.smiles_column)
            if self.id_column:
                custom_data_cols.append(self.id_column)

        figure = go.Figure()

        # Line colors adapt to dark/light theme
        line_color = "rgba(200, 200, 200, 0.5)" if self.theme_manager.dark_mode() else "rgba(0, 0, 0, 0.5)"
        boundary_color = "rgba(200, 200, 200, 0.4)" if self.theme_manager.dark_mode() else "rgba(0, 0, 0, 0.4)"

        # Add data point traces first (so lines render on top)
        self._add_data_traces(figure, df, px_arr, py_arr, color_col, class_labels, custom_data_cols, mask)

        # Triangle border (on top of data points)
        figure.add_trace(go.Scattergl(
            x=[0, 1, 0.5, 0], y=[0, 0, h, 0], mode="lines",
            line=dict(color=line_color, width=4), showlegend=False, hoverinfo="skip",
        ))

        # Decision boundary lines: center (1/3, 1/3, 1/3) to each edge midpoint
        cx, cy = self._project(1/3, 1/3, 1/3)
        for edge in [(0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]:
            ex, ey = self._project(*edge)
            figure.add_trace(go.Scattergl(
                x=[cx, ex], y=[cy, ey], mode="lines",
                line=dict(color=boundary_color, width=2, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))

        # Vertex labels + title annotation below the triangle
        pad = 0.03
        annotations = [
            dict(x=-pad, y=-pad, text=class_labels[0], showarrow=False, font=dict(size=18)),
            dict(x=0.5, y=h + pad, text=class_labels[1], showarrow=False, font=dict(size=18)),
            dict(x=1 + pad, y=-pad, text=class_labels[2], showarrow=False, font=dict(size=18)),
            dict(x=0.5, y=-0.05, text="<b>Prediction Probabilities</b>", showarrow=False,
                 font=dict(size=16), xanchor="center", yanchor="top"),
        ]
        figure.update_layout(
            xaxis=dict(visible=False, range=[-0.1, 1.1], scaleanchor="y", scaleratio=1, constrain="domain"),
            yaxis=dict(visible=False, range=[-0.12, h + 0.1], constrain="domain"),
            plot_bgcolor=self.theme_manager.background(),
            paper_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 0},
            showlegend=True, dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},
            annotations=annotations,
        )
        return figure

    def _add_data_traces(self, figure, df, px_arr, py_arr, color_col, class_labels, custom_data_cols, mask=None):
        """Add data point traces to the figure (normal or selection-style).

        Args:
            figure (go.Figure): The Plotly figure to add traces to.
            df (pd.DataFrame): The dataframe containing prediction data.
            px_arr (np.ndarray): Projected x coordinates.
            py_arr (np.ndarray): Projected y coordinates.
            color_col (str): The column to use for coloring points.
            class_labels (list): The three class labels.
            custom_data_cols (list): Columns to include in customdata for hover.
            mask (pd.Series): Optional boolean mask. None = normal mode, Series = selection mode.
        """
        # In selection mode, add dimmed background points first
        if mask is not None:
            bg_mask = ~mask
            if bg_mask.any():
                figure.add_trace(go.Scattergl(
                    x=px_arr[bg_mask.values], y=py_arr[bg_mask.values],
                    mode="markers", hoverinfo="skip", marker=_DIMMED_MARKER, showlegend=False,
                ))
            # Use only selected points for coloring
            plot_df = df[mask]
            plot_px, plot_py = px_arr[mask.values], py_arr[mask.values]
        else:
            plot_df = df
            plot_px, plot_py = px_arr, py_arr

        if plot_df.empty:
            return

        # Numeric coloring (e.g., residual)
        if color_col in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[color_col]):
            colorscale = add_alpha_to_first_color(self.theme_manager.colorscale("heatmap"))
            marker = dict(size=15, color=plot_df[color_col], colorscale=colorscale,
                          colorbar=dict(title=color_col, thickness=10), line=_MARKER_LINE)
            # In selection mode, pin colorscale to full dataframe range
            if mask is not None:
                marker["cmin"], marker["cmax"] = df[color_col].min(), df[color_col].max()
            cdata = plot_df[custom_data_cols].values if custom_data_cols else None
            figure.add_trace(go.Scattergl(
                x=plot_px, y=plot_py, mode="markers", hoverinfo="none",
                customdata=cdata, marker=marker, showlegend=False,
            ))

        # Categorical coloring (one trace per category)
        elif color_col in plot_df.columns:
            categories = plot_df[color_col].astype(str).unique().tolist()
            categories = list(class_labels) if set(categories) == set(class_labels) else sorted(categories)
            colors = pex.colors.qualitative.Plotly
            for i, cat in enumerate(categories):
                cat_mask = plot_df[color_col].astype(str) == cat
                cdata = plot_df.loc[cat_mask, custom_data_cols].values if custom_data_cols else None
                figure.add_trace(go.Scattergl(
                    x=plot_px[cat_mask.values], y=plot_py[cat_mask.values],
                    mode="markers", hoverinfo="none", name=cat, customdata=cdata,
                    marker=dict(size=15, color=colors[i % len(colors)], line=_MARKER_LINE),
                ))

    def register_internal_callbacks(self):
        """Register internal callbacks for the plugin."""

        @callback(
            Output(f"{self.component_id}-graph", "figure", allow_duplicate=True),
            Input(f"{self.component_id}-color-dropdown", "value"),
            prevent_initial_call=True,
        )
        def _update_ternary_color(color_value):
            if self.df is None or self.df.empty or not color_value:
                raise PreventUpdate
            return self.create_ternary_plot(self.df, self.class_labels, self.proba_cols, color_value)

        # Clientside callback for circle overlay - runs in browser, no server round trip
        clientside_callback(
            circle_overlay_callback(self._circle_data_uri),
            Output(f"{self.component_id}-overlay", "show"),
            Output(f"{self.component_id}-overlay", "bbox"),
            Output(f"{self.component_id}-overlay", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )

        @callback(
            Output(f"{self.component_id}-molecule-tooltip", "show"),
            Output(f"{self.component_id}-molecule-tooltip", "bbox"),
            Output(f"{self.component_id}-molecule-tooltip", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )
        def _molecule_overlay(hover_data):
            if hover_data is None or not self.has_smiles:
                return False, no_update, no_update

            customdata = hover_data["points"][0].get("customdata")
            if customdata is None:
                return False, no_update, no_update

            if isinstance(customdata, (list, tuple)):
                smiles = customdata[0]
                mol_id = customdata[1] if len(customdata) > 1 and self.id_column else None
            else:
                smiles, mol_id = customdata, None

            mol_width, mol_height = 300, 200
            children = molecule_hover_tooltip(
                smiles, mol_id=mol_id, width=mol_width, height=mol_height, background=self.hover_background
            )

            bbox = hover_data["points"][0]["bbox"]
            cx = (bbox["x0"] + bbox["x1"]) / 2
            cy = (bbox["y0"] + bbox["y1"]) / 2
            adjusted_bbox = {
                "x0": cx + 5, "x1": cx + 5 + mol_width,
                "y0": cy - mol_height - (mol_height + 50), "y1": cy - (mol_height + 50),
            }
            return True, adjusted_bbox, children


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    model = CachedModel("aqsol-mol-class")
    PluginUnitTest(ConfusionTriangle, input_data=model, theme="dark").run()
