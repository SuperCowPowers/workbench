"""An Endpoint Metrics Plots component"""

import math
from dash import dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Workbench Imports
from workbench.web_interface.components.component_interface import ComponentInterface
from workbench.utils.pandas_utils import subplot_positions

# Get local timezone
local_tz = datetime.now().astimezone().tzinfo


class EndpointMetricPlots(ComponentInterface):
    """Endpoint Metrics Component"""

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Endpoint Metrics Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Endpoint Metrics Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, endpoint_details: dict) -> go.Figure:
        """Create a Endpoint Metrics Figure for the numeric columns in the dataframe.
        Args:
            endpoint_details (dict): The model details dictionary
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the Endpoint metrics.
        """

        # Grab the Endpoint metrics from the model details
        metrics_df = endpoint_details.get("endpoint_metrics")
        if metrics_df is None:
            return self.display_text("No Data")

        # Let's convert all the timestamps to local timezone
        metrics_df["timestamps"] = metrics_df["timestamps"].dt.tz_convert(local_tz)

        # Move the timestamps to the index
        metrics_df.set_index("timestamps", inplace=True, drop=True)

        # Compute our subplot layout
        subplot_pos_lookup = subplot_positions(metrics_df)
        num_rows = math.ceil(len(metrics_df.columns) / 2)

        # Create the figure with subplots for each metric
        fig = make_subplots(rows=num_rows, cols=2, subplot_titles=metrics_df.columns, vertical_spacing=0.2)
        for metric in metrics_df.columns:
            row, col = subplot_pos_lookup[metric]
            fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df[metric], fill="toself"), row=row, col=col)

        # Update the figure layout
        fig.update_xaxes(tickfont_size=10)
        fig.update_yaxes(rangemode="tozero", tickfont_size=10)
        fig.update_layout(showlegend=False, margin={"t": 30, "b": 20, "r": 10, "l": 30}, height=400)

        # Return the figure
        return fig


if __name__ == "__main__":
    # This class plots out Endpoint Metrics
    from workbench.core.artifacts.endpoint_core import EndpointCore

    # Grab the endpoint details
    end = EndpointCore("abalone-regression-end-rt")
    end_details = end.details()

    # Instantiate the EndpointMetricPlots class
    endpoint_metric_plots = EndpointMetricPlots()

    # Generate the figure
    fig = endpoint_metric_plots.update_properties(end_details)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
