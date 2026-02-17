"""Confusion Explorer: A compound plugin combining a residual-colored Confusion Matrix + Confusion Triangle."""

from math import log
from dash import dcc, html, callback, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.confusion_triangle import ConfusionTriangle
from workbench.cached.cached_model import CachedModel
from workbench.utils.pandas_utils import max_proba, proba_to_conf, compute_confusion
from workbench.utils.color_utils import sample_colorscale_rgba


def _highlight_shape(x_idx: int, y_idx: int) -> dict:
    """Build a Plotly shape dict for highlighting a confusion matrix cell.

    Color comes from the Plotly template's shapedefaults.line.color, which is
    theme-aware (light color for dark themes, dark color for light themes).

    Args:
        x_idx (int): Column index of the cell.
        y_idx (int): Row index of the cell.

    Returns:
        dict: A Plotly rect shape specification.
    """
    delta = 0.5
    return {
        "type": "rect",
        "name": "highlight",
        "x0": x_idx - delta,
        "x1": x_idx + delta,
        "y0": y_idx - delta,
        "y1": y_idx + delta,
        "line": {"width": 3},
        "layer": "above",
    }


def _cell_shapes(cm, n_classes: int, colorscale: list) -> list[dict]:
    """Build filled rect shapes for each confusion matrix cell.

    Color is based on residual (distance from diagonal), alpha on log(count).

    Args:
        cm (pd.DataFrame): The confusion matrix (rows=actual, columns=predicted, already flipped).
        n_classes (int): Number of classes.
        colorscale (list): Plotly colorscale for residual coloring.

    Returns:
        list (dict): A list of Plotly rect shape dicts.
    """
    # Alpha: 0.1 for zero-count cells, then log-scaled [0.25, 1.0] for non-zero
    nonzero_counts = [c for c in cm.values.flatten() if c > 0]
    max_log = log(max(nonzero_counts) + 1) if nonzero_counts else 1.0

    shapes = []
    delta = 0.47  # slightly smaller than 0.5 to create gaps between cells
    vmax = max(n_classes - 1, 1)
    for i, row in enumerate(cm.index):
        for j, col in enumerate(cm.columns):
            # The cm is flipped (row 0 = last class), so map back to original row index
            orig_row = n_classes - 1 - i
            residual = abs(orig_row - j)
            count = cm.iloc[i, j]
            if count == 0:
                alpha = 0.1
            else:
                alpha = 0.25 + 0.75 * (log(count + 1) / max_log)
            fillcolor = sample_colorscale_rgba(colorscale, residual, vmin=0, vmax=vmax, alpha=alpha)
            shapes.append(
                {
                    "type": "rect",
                    "name": "cell",
                    "x0": j - delta,
                    "x1": j + delta,
                    "y0": i - delta,
                    "y1": i + delta,
                    "fillcolor": fillcolor,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
    return shapes


class ClassConfusionMatrix(PluginInterface):
    """A residual-colored Confusion Matrix with log-count alpha.

    Cell color is based on residual distance from the diagonal (0 = correct,
    higher = worse misclassification). Cell opacity scales with log(count),
    making high-count cells visually prominent and low-count cells faint.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def __init__(self):
        """Initialize the ClassConfusionMatrix plugin class."""
        self.component_id = None
        self.df = None
        self.target_col = None
        self.class_labels = None
        super().__init__()

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            dcc.Graph: The Confusion Matrix Component.
        """
        self.component_id = component_id
        self.container = dcc.Graph(
            id=component_id,
            figure=self.display_text("Waiting for Data..."),
            config={"scrollZoom": False, "doubleClick": "reset", "displayModeBar": False},
            style={"height": "350px", "width": "100%"},
        )

        # Fill in plugin properties
        self.properties = [(self.component_id, "figure")]
        self.signals = [(self.component_id, "clickData")]

        return self.container

    def update_properties(self, df: pd.DataFrame, **kwargs) -> list:
        """Create a residual-colored Confusion Matrix figure.

        Args:
            df (pd.DataFrame): DataFrame containing inference predictions.
            **kwargs (dict):
                - target_col (str): The target column name.
                - class_labels (list): The class labels for the model.

        Returns:
            list: A list containing the updated Plotly figure.
        """
        self.df = df
        self.target_col = kwargs.get("target_col")
        self.class_labels = kwargs.get("class_labels")

        if self.df is None or self.df.empty:
            return [self.display_text("No Data")]

        if self.target_col not in self.df.columns or "prediction" not in self.df.columns:
            return [self.display_text("Missing target/prediction columns")]

        # Build confusion matrix: rows = actual, columns = predicted
        cm = pd.crosstab(self.df[self.target_col], self.df["prediction"], dropna=False)
        cm = cm.reindex(index=self.class_labels, columns=self.class_labels, fill_value=0)

        # Flip for correct orientation (highest class on top)
        cm = cm.iloc[::-1]
        n_classes = len(cm.columns)

        # Labels with embedded index for click handling
        x_labels = [f"{c}:{i}" for i, c in enumerate(cm.columns)]
        y_labels = [f"{c}:{i}" for i, c in enumerate(cm.index)]

        # Build cell shapes: color from residual, alpha from log(count)
        colorscale = self.theme_manager.colorscale("muted_heatmap")
        cell_shapes = _cell_shapes(cm, n_classes, colorscale)

        # Invisible heatmap for click detection (shapes don't generate clickData)
        z_zeros = [[0] * n_classes for _ in range(n_classes)]
        fig = go.Figure(
            data=go.Heatmap(
                z=z_zeros,
                x=x_labels,
                y=y_labels,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                hoverinfo="none",
            )
        )
        fig.update_layout(shapes=cell_shapes)

        # Layout with fixed axis range to prevent shift when highlight shape is added
        pad = 0.6  # slightly larger than the 0.5 shape delta to accommodate line width
        axis_range = [-pad, n_classes - 1 + pad]
        fig.update_layout(
            margin=dict(l=60, r=10, t=0, b=50, pad=0),
            xaxis=dict(title=dict(text="Predicted"), range=axis_range),
            yaxis=dict(title=dict(text="Actual"), range=axis_range),
            title_font_size=14,
        )

        # Configure axes
        fig.update_xaxes(
            tickvals=x_labels,
            ticktext=cm.columns,
            tickangle=30,
            tickfont_size=12,
            title_standoff=20,
            title_font={"size": 18},
            showgrid=False,
        )
        fig.update_yaxes(
            tickvals=y_labels,
            ticktext=cm.index,
            tickfont_size=12,
            title_standoff=20,
            title_font={"size": 18},
            showgrid=False,
        )

        # Annotations: show real count values
        for i, row in enumerate(cm.index):
            for j, col in enumerate(cm.columns):
                value = cm.loc[row, col]
                text_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=text_value,
                    showarrow=False,
                    font_size=18,
                )

        return [fig]

    def set_theme(self, theme: str) -> list:
        """Re-render the confusion matrix when the theme changes.

        Args:
            theme (str): The name of the new theme.

        Returns:
            list: Updated property values.
        """
        if self.df is None:
            return [no_update] * len(self.properties)
        return self.update_properties(self.df, target_col=self.target_col, class_labels=self.class_labels)

    def register_internal_callbacks(self):
        """Register standalone cell highlight callback.

        Note: Not used when embedded in ConfusionExplorer (the explorer's cross-component
        callback handles matrix highlighting instead). Kept for potential standalone use.
        """

        # allow_duplicate: figure also set by page-level callbacks and theme changes
        @callback(
            Output(self.component_id, "figure", allow_duplicate=True),
            Input(self.component_id, "clickData"),
            State(self.component_id, "figure"),
            prevent_initial_call=True,
        )
        def highlight_cm_square(click_data, current_figure):
            """Highlight the selected confusion matrix square."""
            if not click_data or "points" not in click_data:
                return current_figure
            point = click_data["points"][0]
            x_idx = int(point["x"].split(":")[1])
            y_idx = int(point["y"].split(":")[1])
            # Preserve cell shapes, replace only the highlight
            cell_shapes = [s for s in current_figure["layout"].get("shapes", []) if s.get("name") != "highlight"]
            current_figure["layout"]["shapes"] = cell_shapes + [_highlight_shape(x_idx, y_idx)]
            return current_figure


class ConfusionExplorer(PluginInterface):
    """Confusion Explorer: Residual-colored Confusion Matrix + Confusion Triangle with linked selection.

    Both components use residual-based coloring: correct predictions fade out (transparent),
    misclassifications light up (bright). Clicking a matrix cell highlights matching points on
    the triangle while dimming non-matching points. Clicking the same cell again resets to the
    full view.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ConfusionExplorer plugin class."""
        self.component_id = None
        self.model = None
        self.inference_run = None
        self.df = None
        self.class_labels = None
        self.target_col = None
        self.proba_cols = None
        self.min_conf = 0.0  # Lower bound for confidence slider (0 = random guess)
        self.confidence_range = None  # [lo, hi] from slider; None = no filtering
        self.matrix = ClassConfusionMatrix()
        self.triangle = ConfusionTriangle()
        self.triangle.default_color = "residual"  # Sync residual coloring between matrix and triangle
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a compound component with Confusion Matrix and Confusion Triangle side by side.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A container with both sub-components.
        """
        self.component_id = component_id

        # Create sub-components with namespaced IDs
        matrix_component = self.matrix.create_component(f"{component_id}-matrix")
        triangle_component = self.triangle.create_component(f"{component_id}-triangle")

        # Aggregate properties from both sub-plugins + slider
        self.properties = (
            list(self.matrix.properties)
            + list(self.triangle.properties)
            + [
                (f"{component_id}-confidence-slider", "marks"),
                (f"{component_id}-confidence-slider", "min"),
                (f"{component_id}-confidence-slider", "value"),
            ]
        )

        # Expose signals from both sub-plugins
        self.signals = list(self.matrix.signals) + list(self.triangle.signals)

        # Confidence slider (marks populated in update_properties)
        slider = dcc.RangeSlider(
            id=f"{component_id}-confidence-slider",
            min=0.0,
            max=1.0,
            step=0.01,
            value=[0.0, 1.0],
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            className="confidence-slider",
        )

        # Confidence slider row: "Confidence" label + slider
        slider_row = html.Div(
            [
                html.Label(
                    "Confidence",
                    style={"fontWeight": "bold", "fontSize": "12px", "whiteSpace": "nowrap"},
                ),
                html.Div(slider, style={"flex": 1, "minWidth": "150px", "marginTop": "20px", "marginBottom": "0px"}),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "padding": "5px",
            },
        )

        return html.Div(
            id=component_id,
            children=[
                dcc.Store(id=f"{component_id}-selected-cell", data=None),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Span(
                                    "Confusion Explorer",
                                    style={
                                        "fontSize": "18px",
                                        "fontWeight": "bold",
                                        "padding": "10px 0px 0px 10px",
                                    },
                                ),
                                slider_row,
                                matrix_component,
                            ],
                            width=5,
                            style={"display": "flex", "flexDirection": "column"},
                        ),
                        dbc.Col(triangle_component, width=7, style={"paddingLeft": "0"}),
                    ],
                ),
            ],
        )

    def _child_kwargs(self) -> dict:
        """Build kwargs dict for passing metadata to child plugins.

        Returns:
            dict: Keyword arguments containing class_labels, target_col, and proba_cols.
        """
        return dict(class_labels=self.class_labels, target_col=self.target_col, proba_cols=self.proba_cols)

    def _get_filtered_df(self) -> pd.DataFrame:
        """Return the dataframe filtered by the current confidence range.

        Returns:
            pd.DataFrame: Filtered dataframe (or full dataframe if no filtering needed).
        """
        if self.confidence_range is None or self.df is None:
            return self.df
        lo, hi = self.confidence_range
        if lo <= self.min_conf and hi >= 1.0:
            return self.df
        return self.df[(self.df["confidence"] >= lo) & (self.df["confidence"] <= hi)].reset_index(drop=True)

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Fetch data from the model and update both sub-plugins with the dataframe.

        Args:
            model (CachedModel): Workbench instantiated CachedModel object.
            **kwargs (dict):
                - inference_run (str): Inference capture name (default: "full_cross_fold").

        Returns:
            list: Combined property values from both sub-plugins + slider props.
        """
        self.model = model
        self.inference_run = kwargs.get("inference_run", "full_cross_fold")

        # Fetch data once — parent owns the dataframe
        self.class_labels = model.class_labels()
        self.target_col = model.target()
        self.proba_cols = [f"{label}_proba" for label in self.class_labels] if self.class_labels else []

        self.df = model.get_inference_predictions(self.inference_run)
        if self.df is not None and not self.df.empty:
            # Drop rows with NaN in probability or residual columns (parent responsibility)
            drop_cols = self.proba_cols + (["residual"] if "residual" in self.df.columns else [])
            drop_cols = [c for c in drop_cols if c in self.df.columns]
            if drop_cols:
                self.df = self.df.dropna(subset=drop_cols).reset_index(drop=True)

            # Compute derived columns for filtering/coloring
            if "confidence" not in self.df.columns:
                max_proba(self.df)
                proba_to_conf(self.df)
            compute_confusion(self.df)

        # Slider parameters: confidence ranges from 0 (random guess) to 1 (certain)
        self.min_conf = 0.0
        self.confidence_range = [0.0, 1.0]

        # Fixed 0-to-1 scale marks for the confidence slider
        slider_marks = {round(i * 0.1, 1): str(round(i * 0.1, 1)) for i in range(11)}

        # Pass dataframe + metadata to both children
        child_kwargs = self._child_kwargs()
        matrix_props = self.matrix.update_properties(self.df, **child_kwargs)
        triangle_props = self.triangle.update_properties(self.df, **child_kwargs)

        # Return: matrix props + triangle props + slider props (marks, min, value)
        return matrix_props + triangle_props + [slider_marks, self.min_conf, [self.min_conf, 1.0]]

    def set_theme(self, theme: str) -> list:
        """Re-render both sub-plugins when the theme changes.

        Args:
            theme (str): The name of the new theme.

        Returns:
            list: Updated property values.
        """
        if self.df is None:
            return [no_update] * len(self.properties)

        # Re-render children with the (potentially filtered) dataframe
        filtered_df = self._get_filtered_df()
        child_kwargs = self._child_kwargs()
        matrix_props = self.matrix.update_properties(filtered_df, **child_kwargs)
        triangle_props = self.triangle.update_properties(filtered_df, **child_kwargs)

        # Slider doesn't change on theme switch
        return matrix_props + triangle_props + [no_update, no_update, no_update]

    def register_internal_callbacks(self):
        """Register callbacks for both sub-plugins and the cross-component linkage."""

        # Register the triangle's internal callbacks (color dropdown, hover tooltips)
        self.triangle.register_internal_callbacks()

        # NOTE: We skip self.matrix.register_internal_callbacks() because the matrix
        # highlight is handled by the cross-component callback below (both would
        # try to output to the same matrix figure property from the same clickData input).

        # Cross-component callback: clicking a matrix cell highlights matching points on the triangle
        # Clicking the same cell again (toggle) resets to the full view
        # allow_duplicate: triangle figure also set by color dropdown and confidence slider;
        #                   matrix figure also set by page-level callbacks and confidence slider
        @callback(
            Output(f"{self.component_id}-triangle-graph", "figure", allow_duplicate=True),
            Output(f"{self.component_id}-selected-cell", "data"),
            Output(f"{self.component_id}-matrix", "figure", allow_duplicate=True),
            Output(f"{self.component_id}-matrix", "clickData"),
            Input(f"{self.component_id}-matrix", "clickData"),
            State(f"{self.component_id}-selected-cell", "data"),
            State(f"{self.component_id}-matrix", "figure"),
            State(f"{self.component_id}-triangle-color-dropdown", "value"),
            prevent_initial_call=True,
        )
        def _select_triangle_from_matrix(click_data, prev_cell, matrix_figure, current_color):
            """Highlight matching points on the triangle for the clicked matrix cell (toggle to reset)."""
            if not click_data or "points" not in click_data:
                raise PreventUpdate

            tri = self.triangle
            if tri.df is None or tri.df.empty:
                raise PreventUpdate

            color_col = current_color if current_color else tri.active_color_col

            # Parse the clicked cell: x = predicted class, y = actual class
            point = click_data["points"][0]
            x_label = point["x"].split(":")[0]  # "label:index" format
            y_label = point["y"].split(":")[0]
            cell_key = f"{x_label}|{y_label}"

            # Preserve cell shapes, strip any existing highlight
            cell_shapes = [s for s in matrix_figure["layout"].get("shapes", []) if s.get("name") != "highlight"]

            # Toggle: if clicking the same cell again, reset to full view
            if prev_cell == cell_key:
                full_fig = tri.create_ternary_plot(tri.df, tri.class_labels, tri.proba_cols, color_col)
                matrix_figure["layout"]["shapes"] = cell_shapes
                return full_fig, None, matrix_figure, None

            # Build selection mask: actual class matches y_label AND predicted class matches x_label
            mask = (tri.df[tri.target_col].astype(str) == y_label) & (tri.df["prediction"].astype(str) == x_label)
            sel_fig = tri.create_ternary_plot(tri.df, tri.class_labels, tri.proba_cols, color_col, mask=mask)

            # Apply highlight rectangle on the matrix cell
            x_idx = int(point["x"].split(":")[1])
            y_idx = int(point["y"].split(":")[1])
            matrix_figure["layout"]["shapes"] = cell_shapes + [_highlight_shape(x_idx, y_idx)]

            # Clear clickData so re-clicking the same cell fires this callback again
            return sel_fig, cell_key, matrix_figure, None

        # Confidence slider callback: filter both children by confidence range
        # Preserves the active matrix cell selection if one exists
        # allow_duplicate: all outputs shared with page-level callbacks or matrix-click callback above
        @callback(
            Output(f"{self.component_id}-matrix", "figure", allow_duplicate=True),
            Output(f"{self.component_id}-triangle-graph", "figure", allow_duplicate=True),
            Output(f"{self.component_id}-triangle-color-dropdown", "options", allow_duplicate=True),
            Output(f"{self.component_id}-triangle-color-dropdown", "value", allow_duplicate=True),
            Output(f"{self.component_id}-selected-cell", "data", allow_duplicate=True),
            Input(f"{self.component_id}-confidence-slider", "value"),
            State(f"{self.component_id}-selected-cell", "data"),
            State(f"{self.component_id}-triangle-color-dropdown", "value"),
            prevent_initial_call=True,
        )
        def _filter_by_confidence(slider_value, prev_cell, current_color):
            """Filter both children by confidence range, preserving matrix cell selection."""
            if slider_value is None or self.df is None:
                raise PreventUpdate

            lo, hi = slider_value
            self.confidence_range = [lo, hi]
            filtered_df = self._get_filtered_df()

            if filtered_df.empty:
                empty_fig = self.display_text("No data in range")
                return [empty_fig, empty_fig, [], None, None]

            # Update children with filtered data
            child_kwargs = self._child_kwargs()
            matrix_props = self.matrix.update_properties(filtered_df, **child_kwargs)
            triangle_props = self.triangle.update_properties(filtered_df, **child_kwargs)

            # Re-render the triangle with the user's current color selection
            tri = self.triangle
            color_col = current_color if current_color else tri.active_color_col
            mask = None

            # If a matrix cell was selected, re-apply the selection on the filtered data
            if prev_cell:
                x_label, y_label = prev_cell.split("|")
                mask = (tri.df[tri.target_col].astype(str) == y_label) & (tri.df["prediction"].astype(str) == x_label)

                # Re-apply the highlight rectangle on the matrix (cell shapes already present from update_properties)
                str_labels = [str(c) for c in (self.class_labels or [])]
                if x_label in str_labels and y_label in str_labels:
                    x_idx = str_labels.index(x_label)
                    y_idx = [str(c) for c in reversed(self.class_labels)].index(y_label)
                    existing = list(matrix_props[0].layout.shapes or [])
                    matrix_props[0].update_layout(shapes=existing + [_highlight_shape(x_idx, y_idx)])

            # Re-create figure with current color, preserve dropdown selection
            triangle_props[0] = tri.create_ternary_plot(tri.df, tri.class_labels, tri.proba_cols, color_col, mask=mask)
            triangle_props[1] = no_update  # options
            triangle_props[2] = no_update  # value

            # Return: matrix figure + triangle (figure, options, value) + preserve cell
            return matrix_props + triangle_props + [prev_cell]


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Use the aqsol-mol-class model (3-class classifier)
    model = CachedModel("aqsol-mol-class")
    # model = CachedModel("caco2-pappab-class-pytorch-1-dt")
    PluginUnitTest(ConfusionExplorer, input_data=model, theme="dark", inference_run="auto_inference").run()
