"""A Model Plot plugin that displays the appropriate visualization based on model type.

For classifiers: Shows a Confusion Matrix
For regressors: Shows a Scatter Plot of predictions vs actuals
"""

from dash import html, no_update

# Workbench Imports
from workbench.api import ModelType
from workbench.cached.cached_model import CachedModel
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.confusion_matrix import ConfusionMatrix
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot


class ModelPlot(PluginInterface):
    """Model Plot Plugin - switches between ConfusionMatrix and ScatterPlot based on model type."""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ModelPlot plugin class"""
        self.component_id = None
        self.model = None
        self.inference_run = None

        # Internal plugins
        self.scatter_plot = ScatterPlot()
        self.confusion_matrix = ConfusionMatrix()

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a container with both ScatterPlot and ConfusionMatrix components.

        Args:
            component_id (str): The ID of the web component

        Returns:
            html.Div: Container with both plot types (one hidden based on model type)
        """
        self.component_id = component_id

        # Create internal components
        scatter_component = self.scatter_plot.create_component(f"{component_id}-scatter")
        confusion_component = self.confusion_matrix.create_component(f"{component_id}-confusion")

        # Build properties list: visibility styles + scatter props + confusion props
        self.properties = [
            (f"{component_id}-scatter-container", "style"),
            (f"{component_id}-confusion-container", "style"),
        ]
        self.properties.extend(self.scatter_plot.properties)
        self.properties.extend(self.confusion_matrix.properties)

        # Aggregate signals from both plugins
        self.signals = self.scatter_plot.signals + self.confusion_matrix.signals

        # Create container with both components
        # Show scatter plot by default (will display "Waiting for Data..." until model loads)
        return html.Div(
            id=component_id,
            children=[
                html.Div(
                    scatter_component,
                    id=f"{component_id}-scatter-container",
                    style={"display": "block"},
                ),
                html.Div(
                    confusion_component,
                    id=f"{component_id}-confusion-container",
                    style={"display": "none"},
                ),
            ],
        )

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Update the plot based on model type.

        Args:
            model (CachedModel): The model to visualize
            **kwargs:
                - inference_run (str): Inference capture name (default: "auto_inference")

        Returns:
            list: Property values [scatter_style, confusion_style, ...scatter_props, ...confusion_props]
        """
        # Cache for theme re-rendering
        self.model = model
        self.inference_run = kwargs.get("inference_run", "full_cross_fold")

        # Determine model type and set visibility
        is_classifier = model.model_type == ModelType.CLASSIFIER
        scatter_style = {"display": "none"} if is_classifier else {"display": "block"}
        confusion_style = {"display": "block"} if is_classifier else {"display": "none"}

        if is_classifier:
            # Update ConfusionMatrix, no_update for ScatterPlot
            cm_props = self.confusion_matrix.update_properties(model, inference_run=self.inference_run)
            scatter_props = [no_update] * len(self.scatter_plot.properties)
        else:
            # Update ScatterPlot with regression data
            df = model.get_inference_predictions(self.inference_run)
            if df is None:
                # Still update visibility styles, but no_update for plugin properties
                scatter_props = [no_update] * len(self.scatter_plot.properties)
                cm_props = [no_update] * len(self.confusion_matrix.properties)
                return [scatter_style, confusion_style] + scatter_props + cm_props

            # Get target column for the x-axis
            target = model.target()
            if isinstance(target, list):
                target = next((t for t in target if t in self.inference_run), target[0])

            # Check if "confidence" column exists for coloring
            color_col = "confidence" if "confidence" in df.columns else "prediction"

            scatter_props = self.scatter_plot.update_properties(
                df,
                x=target,
                y="prediction",
                color=color_col,
                regression_line=True,
            )
            cm_props = [no_update] * len(self.confusion_matrix.properties)

        return [scatter_style, confusion_style] + scatter_props + cm_props

    def set_theme(self, theme: str) -> list:
        """Re-render the appropriate plot when the theme changes."""
        if self.model is None:
            return [no_update] * len(self.properties)

        # Just call update_properties which will re-render the right plot
        return self.update_properties(self.model, inference_run=self.inference_run)

    def register_internal_callbacks(self):
        """Register internal callbacks for both sub-plugins."""
        self.scatter_plot.register_internal_callbacks()
        self.confusion_matrix.register_internal_callbacks()


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Test with a classifier (shows Confusion Matrix)
    classifier_model = CachedModel("wine-classification")
    PluginUnitTest(ModelPlot, input_data=classifier_model, theme="dark").run()

    # Test with a regressor (shows Scatter Plot)
    regressor_model = CachedModel("abalone-regression")
    PluginUnitTest(ModelPlot, input_data=regressor_model, theme="dark").run()
