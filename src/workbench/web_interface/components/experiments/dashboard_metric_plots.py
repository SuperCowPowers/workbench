"""An Dashboard Metrics Plots component"""

import math

import pandas as pd
from dash import dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Workbench Imports
from workbench.web_interface.components.component_interface import ComponentInterface
from workbench.utils.pandas_utils import subplot_positions

# Get local timezone
local_tz = datetime.now().astimezone().tzinfo


class DashboardMetricPlots(ComponentInterface):
    """Dashboard Metrics Component"""

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Dashboard Metrics Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Dashboard Metrics Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Create a Dashboard Metric Plots Figure.
        Args:
            metrics_df (pd.DataFrame): The dashboard metrics dataframe
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the Dashboard metrics.
        """

        # Grab the Dashboard metrics
        if metrics_df is None or metrics_df.empty:
            return self.display_text("No Data")

        # Let's convert all the timestamps to local timezone
        metrics_df["timestamps"] = metrics_df["timestamps"].dt.tz_convert(local_tz)

        # Move the timestamps to the index
        metrics_df.set_index("timestamps", inplace=True, drop=True)

        # Compute our subplot layout
        subplot_pos_lookup = subplot_positions(metrics_df)
        num_rows = math.ceil(len(metrics_df.columns) / 2)

        # Create the figure with subplots for each metric
        fig = make_subplots(rows=num_rows, cols=2, subplot_titles=metrics_df.columns, vertical_spacing=0.15)
        for metric in metrics_df.columns:
            row, col = subplot_pos_lookup[metric]
            fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df[metric], fill="toself"), row=row, col=col)

        # Update the figure layout
        fig.update_xaxes(tickfont_size=10)
        fig.update_yaxes(rangemode="tozero", tickfont_size=10)
        fig.update_layout(showlegend=False, margin={"t": 50, "b": 0, "r": 10, "l": 10}, height=800)

        # Return the figure
        return fig


if __name__ == "__main__":
    # This class plots out Dashboard Metrics
    from workbench.utils.dashboard_metrics import DashboardMetrics

    # Grab our Dashboard Metrics
    dashboard_metrics = DashboardMetrics().get_metrics()

    # Instantiate the DashboardMetricPlots class
    dashboard_metric_plots = DashboardMetricPlots()

    # Generate the figure
    fig = dashboard_metric_plots.update_properties(dashboard_metrics)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
