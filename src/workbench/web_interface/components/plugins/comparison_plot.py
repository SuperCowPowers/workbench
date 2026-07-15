"""A Comparison Plot: a ScatterPlot that overlays predictions from two (or more) models"""

from dash import callback, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go

# Workbench Imports
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

# Opacity for a model whose checkbox is unchecked (dimmed, not hidden)
DIM_OPACITY = 0.1


class ComparisonPlot(ScatterPlot):
    """Overlay the predictions of two models (e.g. champion vs challenger) in one scatter plot.

    Takes the DataFrame shape produced by workbench.utils.model_comparison.prediction_comparison():
    stacked per-model predictions with a 'model' column. Points are colored by model, and a
    checkbox per model dims (rather than hides) that model's points when unchecked.
    """

    def __init__(self):
        """Initialize the Comparison Plot Plugin"""
        self.model_names = []
        self.dim_models = set()
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create the ScatterPlot component plus a per-model dim checklist."""
        container = super().create_component(component_id)

        # Add the model checklist to the controls row (options/values filled by update_properties)
        controls = container.children[1]
        controls.children.append(
            dcc.Checklist(
                id=f"{component_id}-model-toggle",
                options=[],
                value=[],
                style={"marginLeft": "20px", "display": "flex", "alignItems": "center", "gap": "15px"},
            )
        )
        self.properties += [
            (f"{component_id}-model-toggle", "options"),
            (f"{component_id}-model-toggle", "value"),
        ]
        return container

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Update the plot with stacked per-model predictions (requires a 'model' column).

        Defaults tuned for comparison: y=prediction, color=model, diagonal on, prediction
        interval bands off (bands from mixed models would be misleading).
        """
        if "model" not in input_data.columns:
            raise ValueError("ComparisonPlot input data requires a 'model' column (see prediction_comparison)")
        self.model_names = list(input_data["model"].unique())
        self.dim_models = set()

        kwargs.setdefault("y", "prediction")
        kwargs.setdefault("color", "model")
        kwargs.setdefault("regression_line", True)
        kwargs.setdefault("pred_intervals", False)
        properties = super().update_properties(input_data, **kwargs)

        # All models start checked (fully visible)
        options = [{"label": f" {name}", "value": name} for name in self.model_names]
        return properties + [options, self.model_names]

    def create_scatter_plot(self, df: pd.DataFrame, *args, **kwargs) -> go.Figure:
        """Create the scatter plot, then dim any unchecked models' traces."""
        figure = super().create_scatter_plot(df, *args, **kwargs)
        for trace in figure.data:
            if trace.name in self.dim_models:
                trace.marker.opacity = DIM_OPACITY
        return figure

    def register_internal_callbacks(self):
        """Register the ScatterPlot callbacks plus the model dim toggle."""
        super().register_internal_callbacks()

        @callback(
            Output(f"{self.component_id}-graph", "figure", allow_duplicate=True),
            Input(f"{self.component_id}-model-toggle", "value"),
            State(f"{self.component_id}-x-dropdown", "value"),
            State(f"{self.component_id}-y-dropdown", "value"),
            State(f"{self.component_id}-color-dropdown", "value"),
            State(f"{self.component_id}-regression-line", "value"),
            State(f"{self.component_id}-pred-intervals", "value"),
            prevent_initial_call=True,
        )
        def _dim_unchecked_models(checked, x_value, y_value, color_value, regression_line, pred_intervals):
            if self.df is None or self.df.empty or not x_value or not y_value or not color_value:
                raise PreventUpdate
            self.dim_models = set(self.model_names) - set(checked or [])
            return self.create_scatter_plot(self.df, x_value, y_value, color_value, regression_line, pred_intervals)


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.cached.cached_model import CachedModel
    from workbench.utils.model_comparison import prediction_comparison

    # Champion vs challenger predictions
    champion = CachedModel("aqsol-regression")
    challenger = CachedModel("aqsol-regression-2")
    df = prediction_comparison(champion, challenger, "full_cross_fold")
    print(df.head())

    # Run the Unit Test on the Plugin
    PluginUnitTest(ComparisonPlot, input_data=df, theme="dark").run()
