"""SHAP Summary Plot visualization component for XGBoost models"""

from dash import dcc, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List

# Workbench Imports
from workbench.cached.cached_model import CachedModel
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.plot_utils import beeswarm_offsets


class ShapSummaryPlot(PluginInterface):
    """SHAP Summary Visualization Component for XGBoost model explanations"""

    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ShapSummaryPlot plugin class"""
        self.component_id = None
        self.model = None  # Store the model for re-rendering on theme change
        super().__init__()

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a SHAP Summary Visualization Component without any data."""
        self.component_id = component_id
        self.container = dcc.Graph(
            id=component_id,
            figure=self.display_text("Waiting for SHAP data..."),
            config={"scrollZoom": False, "doubleClick": "reset", "displayModeBar": False},
        )
        self.properties = [(self.component_id, "figure")]
        self.signals = [(self.component_id, "clickData")]
        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Create a SHAP Summary Plot for feature importance visualization."""
        # Store for re-rendering on theme change
        self.model = model

        # Basic validation
        shap_data = model.shap_data()
        shap_sample_rows = model.shap_sample()
        if shap_data is None or shap_sample_rows is None:
            return [self.display_text("SHAP data not available")]

        # Check if the model is multiclass
        is_multiclass = isinstance(shap_data, dict)
        if is_multiclass:
            class_labels = model.class_labels()
            id_column = shap_data[list(shap_data.keys())[0]].columns[0]
            fig = self._create_multiclass_summary_plot(shap_data, shap_sample_rows, id_column, class_labels)

        # Regression or binary classification
        else:
            id_column = shap_data.columns[0]
            fig = self._create_summary_plot(shap_data, shap_sample_rows, id_column)
        return [fig]

    def _create_summary_plot(self, shap_df: pd.DataFrame, sample_df: pd.DataFrame, id_column: str) -> go.Figure:
        """Create a SHAP summary plot for a single class."""

        # Remove bias column if present
        if "bias" in shap_df.columns:
            shap_df.drop(columns=["bias"], inplace=True)

        # Grab the shap features (all columns except the ID column)
        shap_features = [feature for feature in shap_df.columns if feature != id_column]

        # Right now we are only supporting numeric features
        shap_features = [
            feature for feature in shap_features if feature in sample_df.select_dtypes(include="number").columns
        ]

        # Merge SHAP values with feature values
        merged_df = pd.merge(shap_df, sample_df, on=id_column, how="inner", suffixes=("_shap", ""))

        # Create figure
        fig = go.Figure()

        # Add a zero vertical line for reference
        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=-0.5,
            y1=len(shap_features) - 0.5,
            line=dict(color="gray", width=2),
            layer="below",
        )

        # Add traces for each feature
        for i, feature in enumerate(shap_features):
            feature_shap = f"{feature}_shap"

            # Normalize feature values for this specific feature (0-1 scale)
            feature_vals = merged_df[feature].values
            norm_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals) + 1e-10)

            # Get y positions with beeswarm offsets
            y_jitter = beeswarm_offsets(merged_df[feature_shap]) + i

            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=merged_df[feature_shap],
                    y=y_jitter,
                    mode="markers",
                    name=feature,
                    marker=dict(
                        color=norm_vals,
                        colorbar=(
                            dict(
                                title="Feature Value",  # Add the title
                                title_side="right",
                                tickvals=[0, 1],
                                ticktext=["Low", "High"],
                                thickness=10,
                            )
                            if i == 0
                            else None
                        ),  # Only show colorbar for first feature
                        opacity=1.0,
                        size=8,
                        showscale=(i == 0),  # Only show color scale for first feature
                    ),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=[
                        f"Feature: {feature}<br>SHAP value: {shap:.4f}<br>Feature value: {val:.4f}"
                        for shap, val in zip(merged_df[feature_shap], feature_vals)
                    ],
                )
            )

        # Update layout
        tick_labels = [f[:15] for f in shap_features]  # Truncate labels to 15 characters
        fig.update_layout(
            title="SHAP Summary Plot: Feature Impact",
            xaxis_title="SHAP Value (Impact on Model Output)",
            margin={"t": 50, "b": 60, "r": 0, "l": 0, "pad": 0},
            height=max(400, 50 * len(shap_features)),
            plot_bgcolor=self.theme_manager.background(),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(
                tickvals=list(range(len(shap_features))),
                ticktext=tick_labels,
                autorange="reversed",  # Most important features at top
                showgrid=False,
                zeroline=False,
            ),
        )
        return fig

    def _create_multiclass_summary_plot(
        self, shap_values: Dict[str, pd.DataFrame], sample_df: pd.DataFrame, id_column: str, class_labels: List[str]
    ) -> go.Figure:
        """Create a SHAP summary plot for multiple classes with class selector."""

        # Get list of classes
        class_names = class_labels
        first_class = class_names[0]

        # Create base figure for first class
        main_fig = self._create_summary_plot(shap_values[first_class], sample_df, id_column)

        # Store traces from first class
        first_class_traces = list(range(len(main_fig.data)))
        traces_per_class = len(first_class_traces)

        # Add traces for all other classes (initially invisible)
        for class_name in class_names[1:]:
            # Create figure for this class
            class_fig = self._create_summary_plot(shap_values[class_name], sample_df, id_column)

            # Add all traces from class_fig to main_fig (set to invisible)
            for trace in class_fig.data:
                # Update hover text to include class
                if hasattr(trace, "hovertext"):
                    new_hovertext = []
                    for ht in trace.hovertext:
                        new_hovertext.append(f"Class: {class_name}<br>{ht}")
                    trace.hovertext = new_hovertext

                # Set to invisible and add to main figure
                trace.visible = False
                main_fig.add_trace(trace)

        # Create buttons for class selection
        buttons = []
        for i, class_name in enumerate(class_names):
            # Create visibility list
            total_traces = len(main_fig.data)

            # Set visibility (True for selected class, False for others)
            visibility = [False] * total_traces
            class_traces_start = i * traces_per_class
            class_traces_end = class_traces_start + traces_per_class

            for j in range(class_traces_start, class_traces_end):
                visibility[j] = True

            # Add button for this class
            buttons.append(dict(method="restyle", label=str(class_name), args=[{"visible": visibility}]))

        # Update layout to add dropdown menu with static title
        main_fig.update_layout(
            title="SHAP Summary Plot: Feature Impact",
            updatemenus=[
                {
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.0,
                    "y": 1.15,
                    "xanchor": "right",
                    "yanchor": "top",
                }
            ],
        )
        return main_fig

    def set_theme(self, theme: str) -> list:
        """Re-render the SHAP summary plot when the theme changes."""
        if self.model is None:
            return [no_update] * len(self.properties)
        return self.update_properties(self.model)

    def register_internal_callbacks(self):
        """Register internal callbacks for the plugin."""
        pass  # No internal callbacks needed


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    # model = CachedModel("abalone-regression")
    model = CachedModel("wine-classification")
    # model = CachedModel("test-classification")
    PluginUnitTest(ShapSummaryPlot, input_data=model, theme="light").run()
