"""Confusion Explorer: A compound plugin combining a residual-colored Confusion Matrix + Confusion Triangle."""

from dash import dcc, html, callback, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.confusion_triangle import ConfusionTriangle
from workbench.cached.cached_model import CachedModel


def _residual_z_matrix(n_classes: int) -> list[list[int]]:
    """Build a residual z-matrix for confusion matrix coloring.

    Each cell value is the absolute distance from the diagonal (0, 1, 2, ...).
    Diagonal = 0 (correct), off-diagonal = distance (misclassification severity).

    Args:
        n_classes (int): Number of classes.

    Returns:
        list[list[int]]: The residual z-matrix.
    """
    return [[abs(i - j) for j in range(n_classes)] for i in range(n_classes)]


class ClassConfusionMatrix(PluginInterface):
    """A residual-colored Confusion Matrix designed for the Confusion Explorer.

    Uses a residual z-matrix where diagonal cells (correct predictions) are nearly
    transparent and off-diagonal cells get brighter based on distance from the diagonal.
    This mirrors the per-point residual coloring on the triangle.
    Real count values are shown as annotations.
    """

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ClassConfusionMatrix plugin class."""
        self.component_id = None
        self.model = None
        self.inference_run = None
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
            config={"scrollZoom": False, "doubleClick": "reset"},
        )

        # Fill in plugin properties
        self.properties = [(self.component_id, "figure")]
        self.signals = [(self.component_id, "clickData")]

        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Create a residual-colored Confusion Matrix figure.

        Args:
            model (CachedModel): Workbench instantiated CachedModel object.
            **kwargs (dict):
                - inference_run (str): Inference capture name.

        Returns:
            list: A list containing the updated Plotly figure.
        """
        self.model = model
        self.inference_run = kwargs.get("inference_run", "auto_inference")

        # Retrieve the confusion matrix data (real counts)
        df = model.confusion_matrix(self.inference_run)
        if df is None:
            return [self.display_text("No Data")]

        # Flip for correct orientation (highest class on top)
        df = df.iloc[::-1]
        n_classes = len(df.columns)

        # Labels with embedded index for click handling
        x_labels = [f"{c}:{i}" for i, c in enumerate(df.columns)]
        y_labels = [f"{c}:{i}" for i, c in enumerate(df.index)]

        # Build the residual z-matrix (flipped to match the dataframe orientation)
        z_residual = _residual_z_matrix(n_classes)[::-1]

        # Use the heatmap colorscale (matches the standalone confusion matrix)
        colorscale = self.theme_manager.colorscale("heatmap")

        # Create the heatmap with residual z-values for coloring
        # 0 = diagonal (correct), higher = off-diagonal (errors, bright)
        fig = go.Figure(
            data=go.Heatmap(
                z=z_residual,
                x=x_labels,
                y=y_labels,
                xgap=3,
                ygap=3,
                colorscale=colorscale,
                showscale=False,
                zmin=0,
                zmax=n_classes - 1,
            )
        )

        # Layout
        fig.update_layout(
            margin=dict(l=60, r=0, t=15, b=80, pad=5),
            xaxis=dict(title=dict(text="Predicted")),
            yaxis=dict(title=dict(text="Actual")),
            title_font_size=14,
        )

        # Configure axes
        fig.update_xaxes(
            tickvals=x_labels, ticktext=df.columns, tickangle=30,
            tickfont_size=12, automargin=True, title_standoff=20,
            title_font={"size": 18}, showgrid=False,
        )
        fig.update_yaxes(
            tickvals=y_labels, ticktext=df.index, tickfont_size=12,
            automargin=True, title_standoff=20, title_font={"size": 18},
            showgrid=False,
        )

        # Annotations: show real count values
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                text_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                fig.add_annotation(
                    x=j, y=i, text=text_value,
                    showarrow=False, font_size=14,
                )

        return [fig]

    def set_theme(self, theme: str) -> list:
        """Re-render the confusion matrix when the theme changes.

        Args:
            theme (str): The name of the new theme.

        Returns:
            list: Updated property values.
        """
        if self.model is None:
            return [no_update] * len(self.properties)
        return self.update_properties(self.model, inference_run=self.inference_run)

    def register_internal_callbacks(self):
        """Register the cell highlight callback."""

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

            # Extract clicked point
            point = click_data["points"][0]
            x_index = point["x"]
            y_index = point["y"]

            # Parse the index from labels (e.g., "label:1")
            x_label, x_idx = x_index.split(":")
            y_label, y_idx = y_index.split(":")
            x_idx, y_idx = int(x_idx), int(y_idx)

            # Create highlight rectangle
            delta = 0.5
            highlight_shape = {
                "type": "rect",
                "x0": x_idx - delta,
                "x1": x_idx + delta,
                "y0": y_idx - delta,
                "y1": y_idx + delta,
                "line": {"color": "grey", "width": 2},
                "layer": "above",
            }

            current_figure["layout"]["shapes"] = [highlight_shape]
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

        # Aggregate properties from both sub-plugins
        self.properties = list(self.matrix.properties) + list(self.triangle.properties)

        # Expose signals from both sub-plugins
        self.signals = list(self.matrix.signals) + list(self.triangle.signals)

        return html.Div(
            id=component_id,
            children=[
                # Hidden store for tracking the currently selected matrix cell (toggle support)
                dcc.Store(id=f"{component_id}-selected-cell", data=None),
                html.Div(
                    children=[
                        # Left: Confusion Matrix
                        html.Div(
                            matrix_component,
                            style={"flex": "1", "minWidth": "300px"},
                        ),
                        # Right: Confusion Triangle
                        html.Div(
                            triangle_component,
                            style={"flex": "1", "minWidth": "400px"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "row", "gap": "10px", "width": "100%"},
                ),
            ],
        )

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update both sub-plugins with the same model.

        Args:
            model (CachedModel): Workbench instantiated CachedModel object.
            **kwargs (dict):
                - inference_run (str): Inference capture name (default: "auto_inference").

        Returns:
            list: Combined property values from both sub-plugins.
        """
        self.model = model
        self.inference_run = kwargs.get("inference_run", "auto_inference")

        # Update both sub-plugins
        matrix_props = self.matrix.update_properties(model, **kwargs)
        triangle_props = self.triangle.update_properties(model, **kwargs)

        return matrix_props + triangle_props

    def set_theme(self, theme: str) -> list:
        """Re-render both sub-plugins when the theme changes.

        Args:
            theme (str): The name of the new theme.

        Returns:
            list: Updated property values.
        """
        if self.model is None:
            return [no_update] * len(self.properties)
        return self.update_properties(self.model, inference_run=self.inference_run)

    def register_internal_callbacks(self):
        """Register callbacks for both sub-plugins and the cross-component linkage."""

        # Register the triangle's internal callbacks (color dropdown, hover tooltips)
        self.triangle.register_internal_callbacks()

        # NOTE: We skip self.matrix.register_internal_callbacks() because the matrix
        # highlight is handled by the cross-component callback below (both would
        # try to output to the same matrix figure property from the same clickData input).

        # Cross-component callback: clicking a matrix cell highlights matching points on the triangle
        # Clicking the same cell again (toggle) resets to the full view
        @callback(
            Output(f"{self.component_id}-triangle-graph", "figure", allow_duplicate=True),
            Output(f"{self.component_id}-selected-cell", "data"),
            Output(f"{self.component_id}-matrix", "figure", allow_duplicate=True),
            Input(f"{self.component_id}-matrix", "clickData"),
            State(f"{self.component_id}-selected-cell", "data"),
            State(f"{self.component_id}-matrix", "figure"),
            prevent_initial_call=True,
        )
        def _select_triangle_from_matrix(click_data, prev_cell, matrix_figure):
            """Highlight matching points on the triangle for the clicked matrix cell (toggle to reset)."""
            if not click_data or "points" not in click_data:
                raise PreventUpdate

            tri = self.triangle
            if tri.df is None or tri.df.empty:
                raise PreventUpdate

            # Parse the clicked cell: x = predicted class, y = actual class
            point = click_data["points"][0]
            x_label = point["x"].split(":")[0]  # "label:index" format
            y_label = point["y"].split(":")[0]
            cell_key = f"{x_label}|{y_label}"

            # Use the triangle's default color (residual in explorer mode)
            color_col = tri.default_color or tri.target_col or "prediction"

            # Toggle: if clicking the same cell again, reset to full view
            if prev_cell == cell_key:
                full_figure = tri.create_ternary_plot(tri.df, tri.class_labels, tri.proba_cols, color_col)
                # Clear the highlight rectangle on the matrix
                matrix_figure["layout"]["shapes"] = []
                return full_figure, None, matrix_figure

            # Build selection mask: actual class matches y_label AND predicted class matches x_label
            target_col = tri.target_col
            mask = (
                (tri.df[target_col].astype(str) == y_label)
                & (tri.df["prediction"].astype(str) == x_label)
            )

            # Render with selection-style brushing (all points visible, matching highlighted)
            selection_figure = tri.create_ternary_plot(
                tri.df, tri.class_labels, tri.proba_cols, color_col, mask=mask
            )

            # Apply highlight rectangle on the matrix cell
            x_idx = int(point["x"].split(":")[1])
            y_idx = int(point["y"].split(":")[1])
            delta = 0.5
            matrix_figure["layout"]["shapes"] = [{
                "type": "rect",
                "x0": x_idx - delta, "x1": x_idx + delta,
                "y0": y_idx - delta, "y1": y_idx + delta,
                "line": {"color": "grey", "width": 2},
                "layer": "above",
            }]

            return selection_figure, cell_key, matrix_figure


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Use the aqsol-mol-class model (3-class classifier)
    model = CachedModel("aqsol-mol-class")
    PluginUnitTest(ConfusionExplorer, input_data=model, theme="dark").run()
