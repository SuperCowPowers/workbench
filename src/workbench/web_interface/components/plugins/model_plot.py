"""A Model Plot plugin that displays the appropriate visualization based on model type.

For classifiers: Shows a Confusion Explorer (Confusion Matrix + Confusion Triangle)
For regressors: Shows a Scatter Plot of predictions vs actuals, in the model's
target_transform space by default when it has one
"""

from dash import Input, Output, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.api import ModelType
from workbench.cached.cached_model import CachedModel
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.confusion_explorer import ConfusionExplorer
from workbench.web_interface.components.plugins.scatter_plot import CHECKLIST_STYLE, ScatterPlot
from workbench.utils.target_transform import Identity

_HIDDEN = {"visibility": "hidden", "height": 0, "overflow": "hidden"}


class ModelPlot(PluginInterface):
    """Model Plot Plugin - switches between ConfusionExplorer and ScatterPlot based on model type."""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ModelPlot plugin class"""
        self.component_id = None
        self.model = None
        self.inference_run = None
        self.target = None
        self.transformed = False

        # Internal plugins
        self.scatter_plot = ScatterPlot()
        self.confusion_explorer = ConfusionExplorer()

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a container with both ScatterPlot and ConfusionExplorer components.

        Args:
            component_id (str): The ID of the web component

        Returns:
            html.Div: Container with both plot types (one hidden based on model type)
        """
        self.component_id = component_id

        # Create internal components (the transform toggle sits in the scatter controls row)
        transform_toggle = dcc.Checklist(
            id=f"{component_id}-transformed", options=[], value=[], style={"display": "none"}
        )
        scatter_component = self.scatter_plot.create_component(
            f"{component_id}-scatter", extra_controls=[transform_toggle]
        )
        confusion_component = self.confusion_explorer.create_component(f"{component_id}-confusion")

        # Build properties list: visibility styles + transform toggle + scatter props + confusion props
        self.properties = [
            (f"{component_id}-scatter-container", "style"),
            (f"{component_id}-confusion-container", "style"),
            (f"{component_id}-transformed", "options"),
            (f"{component_id}-transformed", "value"),
            (f"{component_id}-transformed", "style"),
        ]
        self.properties.extend(self.scatter_plot.properties)
        self.properties.extend(self.confusion_explorer.properties)

        # Aggregate signals from both plugins
        self.signals = self.scatter_plot.signals + self.confusion_explorer.signals

        # Create container with both components
        # Show scatter plot by default (will display "Waiting for Data..." until model loads)
        # Note: Use visibility/height instead of display:none so Bootstrap grid computes
        # column widths upfront, preventing Plotly from rendering wide then shrinking.
        return html.Div(
            id=component_id,
            children=[
                html.Div(
                    scatter_component,
                    id=f"{component_id}-scatter-container",
                    style={},
                ),
                html.Div(
                    confusion_component,
                    id=f"{component_id}-confusion-container",
                    style=_HIDDEN,
                ),
            ],
        )

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the plot based on model type.

        Args:
            model (CachedModel): The model to visualize
            **kwargs:
                - inference_run (str): Inference capture name (default: resolved via default_inference_run())

        Returns:
            list: Property values [scatter_style, confusion_style, ...toggle_props, ...scatter_props,
                ...confusion_props]
        """
        # Cache for theme re-rendering
        self.model = model
        self.inference_run = kwargs.get("inference_run") or model.default_inference_run()

        # Determine model type and set visibility
        # Use visibility/height instead of display:none to preserve Bootstrap grid layout
        is_classifier = model.model_type == ModelType.CLASSIFIER
        scatter_style = _HIDDEN if is_classifier else {}
        confusion_style = {} if is_classifier else _HIDDEN
        toggle_props = [[], [], {"display": "none"}]

        if is_classifier:
            # Update ConfusionExplorer, no_update for ScatterPlot
            cm_props = self.confusion_explorer.update_properties(model, inference_run=self.inference_run)
            scatter_props = [no_update] * len(self.scatter_plot.properties)
        else:
            # Get target column for the x-axis
            self.target = model.target()
            if isinstance(self.target, list):
                self.target = next((t for t in self.target if t in self.inference_run), self.target[0])

            # Plot in the model's target_transform space by default when it has one
            transform = model.target_transform()
            if isinstance(transform, dict):
                transform = transform[self.target]
            has_transform = not isinstance(transform, Identity)
            if has_transform:
                options = [{"label": f" {transform!r}", "value": "show"}]
                toggle_props = [options, ["show"], CHECKLIST_STYLE]

            # Update ScatterPlot with regression data
            scatter_props = self._scatter_properties(has_transform)
            cm_props = [no_update] * len(self.confusion_explorer.properties)

        return [scatter_style, confusion_style] + toggle_props + scatter_props + cm_props

    def _scatter_properties(self, transformed: bool) -> list:
        """Render the regression scatter plot, in target_transform space when transformed is set."""
        self.transformed = transformed
        df = self.model.get_inference_predictions(self.inference_run, transformed=transformed)
        if df is None:
            return [no_update] * len(self.scatter_plot.properties)

        # Check if "confidence" column exists for coloring
        color_col = "confidence" if "confidence" in df.columns else "prediction"
        return self.scatter_plot.update_properties(
            df,
            x=self.target,
            y="prediction",
            color=color_col,
            regression_line=True,
        )

    def set_theme(self, theme: str) -> list:
        """Re-render the appropriate plot when the theme changes."""
        if self.model is None:
            return [no_update] * len(self.properties)

        # Just call update_properties which will re-render the right plot
        return self.update_properties(self.model, inference_run=self.inference_run)

    def register_internal_callbacks(self):
        """Register internal callbacks for both sub-plugins and the transform toggle."""
        self.scatter_plot.register_internal_callbacks()
        self.confusion_explorer.register_internal_callbacks()

        # allow_duplicate: the scatter plot is also set by page-level callbacks and theme changes
        @callback(
            [Output(cid, prop, allow_duplicate=True) for cid, prop in self.scatter_plot.properties],
            Input(f"{self.component_id}-transformed", "value"),
            prevent_initial_call=True,
        )
        def _toggle_transformed(value):
            # A page-level update sets the toggle and renders in the same pass
            transformed = bool(value)
            if self.model is None or transformed == self.transformed:
                raise PreventUpdate
            return self._scatter_properties(transformed)


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Test with a classifier (shows Confusion Explorer)
    classifier_model = CachedModel("wine-classification")
    PluginUnitTest(ModelPlot, input_data=classifier_model, theme="dark").run()

    # Test with a regressor (shows Scatter Plot)
    regressor_model = CachedModel("abalone-regression")
    PluginUnitTest(ModelPlot, input_data=regressor_model, theme="dark").run()
