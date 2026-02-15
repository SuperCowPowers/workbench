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
                    style={"height": "500px", "width": "100%"},
                    clear_on_unhover=True,
                ),
                html.Div(
                    [
                        html.Span(
                            "Prediction Probabilities",
                            style={
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "opacity": "0.8",
                            },
                        ),
                        html.Div(style={"flex": 1}),  # Spacer
                        html.Label(
                            "Color",
                            style={
                                "marginRight": "5px",
                                "fontWeight": "bold",
                                "display": "flex",
                                "alignItems": "center",
                            },
                        ),
                        dcc.Dropdown(
                            id=f"{component_id}-color-dropdown",
                            style={"minWidth": "200px"},
                            clearable=False,
                        ),
                    ],
                    style={"padding": "0px 20px 10px 20px", "display": "flex", "alignItems": "center", "gap": "5px"},
                ),
                # Circle overlay tooltip (centered on hovered point)
                dcc.Tooltip(
                    id=f"{component_id}-overlay",
                    background_color="rgba(0,0,0,0)",
                    border_color="rgba(0,0,0,0)",
                    direction="bottom",
                    loading_text="",
                ),
                # Molecule/info tooltip (offset from hovered point)
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

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            model (CachedModel): Workbench instantiated CachedModel object.
            **kwargs (dict):
                - inference_run (str): Inference capture name (default: "auto_inference").

        Returns:
            list: A list of updated property values (figure, color options, color default).
        """
        # Get the background color from the current theme
        self.hover_background = self.theme_manager.background()

        # Cache for theme re-rendering
        self.model = model
        self.inference_run = kwargs.get("inference_run", "auto_inference")

        # Get class labels and validate we have exactly 3
        self.class_labels = model.class_labels()
        if self.class_labels is None or len(self.class_labels) != 3:
            n = len(self.class_labels) if self.class_labels else 0
            return [self.display_text(f"Requires 3-class classifier (got {n} classes)"), [], None]

        # Get inference predictions
        self.df = model.get_inference_predictions(self.inference_run)
        if self.df is None or self.df.empty:
            return [self.display_text("No Prediction Data"), [], None]

        # Build probability column names
        self.proba_cols = [f"{label}_proba" for label in self.class_labels]

        # Verify all proba columns exist
        missing = [col for col in self.proba_cols if col not in self.df.columns]
        if missing:
            return [self.display_text(f"Missing columns: {missing}"), [], None]

        # Detect smiles and id columns for molecule hover rendering
        self.smiles_column = next((col for col in self.df.columns if col.lower() == "smiles"), None)
        self.id_column = next((col for col in self.df.columns if col.lower() == "id"), self.df.columns[0])
        self.has_smiles = self.smiles_column is not None

        # Build color dropdown options
        target_col = model.target()
        self.target_col = target_col
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        cat_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        cat_columns = [col for col in cat_columns if self.df[col].astype(str).nunique() < 20]
        color_columns = numeric_columns + cat_columns

        # Use default_color if set by parent (e.g., ConfusionExplorer sets "residual")
        if self.default_color and self.default_color in color_columns:
            color_default = self.default_color
        else:
            color_default = target_col if target_col in color_columns else color_columns[0] if color_columns else None
        color_options = [{"label": col, "value": col} for col in color_columns]

        # Create the ternary plot
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

        Vertices: bottom-left (0,0) = low, top (0.5, sqrt(3)/2) = mid, bottom-right (1,0) = high.

        Args:
            low (float | np.ndarray): Probability for the lowest class.
            mid (float | np.ndarray): Probability for the middle class.
            high (float | np.ndarray): Probability for the highest class.

        Returns:
            tuple: (x, y) cartesian coordinates.
        """
        h = np.sqrt(3) / 2
        x = high + mid * 0.5
        y = mid * h
        return x, y

    def create_ternary_plot(
        self,
        df: pd.DataFrame,
        class_labels: list,
        proba_cols: list,
        color_col: str,
        mask: pd.Series = None,
    ) -> go.Figure:
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
        low_col, mid_col, high_col = proba_cols
        h = np.sqrt(3) / 2

        # Project all data points to 2D
        px_arr, py_arr = self._project(df[low_col].values, df[mid_col].values, df[high_col].values)

        # Build customdata columns for molecule hover
        custom_data_cols = []
        if self.has_smiles:
            custom_data_cols.append(self.smiles_column)
            if self.id_column:
                custom_data_cols.append(self.id_column)

        figure = go.Figure()

        # Draw the triangle border
        tri_x = [0, 1, 0.5, 0]
        tri_y = [0, 0, h, 0]
        figure.add_trace(
            go.Scattergl(
                x=tri_x, y=tri_y, mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.4)", width=1),
                showlegend=False, hoverinfo="skip",
            )
        )

        # Decision boundary lines: center (1/3, 1/3, 1/3) to each edge midpoint
        cx, cy = self._project(1/3, 1/3, 1/3)
        boundary_color = "rgba(255, 255, 255, 0.3)"
        for edge in [(0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]:
            ex, ey = self._project(*edge)
            figure.add_trace(
                go.Scattergl(
                    x=[cx, ex], y=[cy, ey], mode="lines",
                    line=dict(color=boundary_color, width=1, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                )
            )

        # --- Add data point traces ---
        if mask is not None:
            # Selection mode: dim non-matching, highlight matching
            self._add_selection_traces(figure, df, px_arr, py_arr, color_col, class_labels, custom_data_cols, mask)
        else:
            # Normal mode: all points with full coloring
            self._add_color_traces(figure, df, px_arr, py_arr, color_col, class_labels, custom_data_cols)

        # Vertex labels and layout
        bg_color = self.theme_manager.background()
        pad = 0.03
        annotations = [
            dict(x=0 - pad, y=0 - pad, text=class_labels[0], showarrow=False, font=dict(size=14)),
            dict(x=0.5, y=h + pad, text=class_labels[1], showarrow=False, font=dict(size=14)),
            dict(x=1 + pad, y=0 - pad, text=class_labels[2], showarrow=False, font=dict(size=14)),
        ]
        figure.update_layout(
            xaxis=dict(visible=False, range=[-0.1, 1.1], scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False, range=[-0.1, h + 0.1]),
            plot_bgcolor=bg_color,
            paper_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 0},
            showlegend=True,
            dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},
            annotations=annotations,
        )

        return figure

    def _add_color_traces(self, figure, df, px_arr, py_arr, color_col, class_labels, custom_data_cols):
        """Add colored data point traces to the figure.

        Args:
            figure (go.Figure): The Plotly figure to add traces to.
            df (pd.DataFrame): The dataframe containing prediction data.
            px_arr (np.ndarray): Projected x coordinates.
            py_arr (np.ndarray): Projected y coordinates.
            color_col (str): The column to use for coloring points.
            class_labels (list): The three class labels.
            custom_data_cols (list): Columns to include in customdata for hover.
        """
        if color_col in df.columns and pd.api.types.is_numeric_dtype(df[color_col]):
            # Numeric coloring (e.g., residual)
            colorscale = self.theme_manager.colorscale("heatmap")
            figure.add_trace(
                go.Scattergl(
                    x=px_arr, y=py_arr, mode="markers", hoverinfo="none",
                    customdata=df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=15, color=df[color_col],
                        colorscale=colorscale,
                        colorbar=dict(title=color_col, thickness=10),
                        line=dict(color="rgba(0,0,0,0.25)", width=1),
                    ),
                    showlegend=False,
                )
            )

        elif color_col in df.columns:
            # Categorical coloring (one trace per category)
            categories = df[color_col].astype(str).unique().tolist()
            categories = list(class_labels) if set(categories) == set(class_labels) else sorted(categories)
            discrete_colors = pex.colors.qualitative.Plotly

            for i, cat in enumerate(categories):
                cat_mask = df[color_col].astype(str) == cat
                figure.add_trace(
                    go.Scattergl(
                        x=px_arr[cat_mask.values], y=py_arr[cat_mask.values],
                        mode="markers", hoverinfo="none", name=cat,
                        customdata=df.loc[cat_mask, custom_data_cols] if custom_data_cols else None,
                        marker=dict(
                            size=15, color=discrete_colors[i % len(discrete_colors)],
                            line=dict(color="rgba(0,0,0,0.25)", width=1),
                        ),
                    )
                )

    def _add_selection_traces(self, figure, df, px_arr, py_arr, color_col, class_labels, custom_data_cols, mask):
        """Add selection-style traces: dimmed background + highlighted foreground.

        Args:
            figure (go.Figure): The Plotly figure to add traces to.
            df (pd.DataFrame): The full dataframe containing prediction data.
            px_arr (np.ndarray): Projected x coordinates.
            py_arr (np.ndarray): Projected y coordinates.
            color_col (str): The column to use for coloring highlighted points.
            class_labels (list): The three class labels.
            custom_data_cols (list): Columns to include in customdata for hover.
            mask (pd.Series): Boolean mask where True = selected/highlighted.
        """
        # Dimmed background points (non-selected)
        bg_mask = ~mask
        if bg_mask.any():
            figure.add_trace(
                go.Scattergl(
                    x=px_arr[bg_mask.values], y=py_arr[bg_mask.values],
                    mode="markers", hoverinfo="none",
                    marker=dict(
                        size=10, color="rgba(128, 128, 128, 0.15)",
                        line=dict(color="rgba(0,0,0,0.05)", width=1),
                    ),
                    showlegend=False,
                )
            )

        # Highlighted points (selected) — use full dataframe color range for consistent colorscale
        fg_df = df[mask]
        if fg_df.empty:
            return

        fg_px, fg_py = px_arr[mask.values], py_arr[mask.values]

        if color_col in fg_df.columns and pd.api.types.is_numeric_dtype(fg_df[color_col]):
            colorscale = self.theme_manager.colorscale("heatmap")
            figure.add_trace(
                go.Scattergl(
                    x=fg_px, y=fg_py, mode="markers", hoverinfo="none",
                    customdata=fg_df[custom_data_cols].values if custom_data_cols else None,
                    marker=dict(
                        size=15, color=fg_df[color_col],
                        colorscale=colorscale, cmin=df[color_col].min(), cmax=df[color_col].max(),
                        colorbar=dict(title=color_col, thickness=10),
                        line=dict(color="rgba(0,0,0,0.25)", width=1),
                    ),
                    showlegend=False,
                )
            )
        elif color_col in fg_df.columns:
            categories = fg_df[color_col].astype(str).unique().tolist()
            categories = list(class_labels) if set(categories) == set(class_labels) else sorted(categories)
            discrete_colors = pex.colors.qualitative.Plotly

            for i, cat in enumerate(categories):
                cat_mask = fg_df[color_col].astype(str) == cat
                figure.add_trace(
                    go.Scattergl(
                        x=fg_px[cat_mask.values], y=fg_py[cat_mask.values],
                        mode="markers", hoverinfo="none", name=cat,
                        customdata=fg_df.loc[cat_mask, custom_data_cols].values if custom_data_cols else None,
                        marker=dict(
                            size=15, color=discrete_colors[i % len(discrete_colors)],
                            line=dict(color="rgba(0,0,0,0.25)", width=1),
                        ),
                    )
                )

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""

        @callback(
            Output(f"{self.component_id}-graph", "figure", allow_duplicate=True),
            Input(f"{self.component_id}-color-dropdown", "value"),
            prevent_initial_call=True,
        )
        def _update_ternary_color(color_value):
            """Update the ternary plot when the color dropdown changes."""
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
            """Show molecule tooltip when smiles data is available."""
            if hover_data is None or not self.has_smiles:
                return False, no_update, no_update

            # Extract customdata (contains smiles and id)
            customdata = hover_data["points"][0].get("customdata")
            if customdata is None:
                return False, no_update, no_update

            # SMILES is the first element, ID is second (if available)
            if isinstance(customdata, (list, tuple)):
                smiles = customdata[0]
                mol_id = customdata[1] if len(customdata) > 1 and self.id_column else None
            else:
                smiles = customdata
                mol_id = None

            # Generate molecule tooltip (use cached background color)
            mol_width, mol_height = 300, 200
            children = molecule_hover_tooltip(
                smiles, mol_id=mol_id, width=mol_width, height=mol_height, background=self.hover_background
            )

            # Position molecule tooltip above and slightly right of the point
            bbox = hover_data["points"][0]["bbox"]
            center_x = (bbox["x0"] + bbox["x1"]) / 2
            center_y = (bbox["y0"] + bbox["y1"]) / 2
            x_offset = 5
            y_offset = mol_height + 50

            adjusted_bbox = {
                "x0": center_x + x_offset,
                "x1": center_x + x_offset + mol_width,
                "y0": center_y - mol_height - y_offset,
                "y1": center_y - y_offset,
            }
            return True, adjusted_bbox, children


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Use the aqsol-mol-class model (3-class classifier)
    model = CachedModel("aqsol-mol-class")
    PluginUnitTest(ConfusionTriangle, input_data=model, theme="dark").run()
