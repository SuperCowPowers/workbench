"""A proximity circle visualization component for comparing a query target to its neighbors"""

from dash import dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager


class ProximityMiniGraph(PluginInterface):
    """Proximity Visualization Component for comparing a target and its nearest neighbors"""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def __init__(self):
        """Initialize the ProximityMiniGraph plugin class"""
        self.component_id = None
        self.theme_manager = ThemeManager()
        self.colorscale = self.theme_manager.colorscale()
        super().__init__()

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Proximity Visualization Component without any data."""
        self.component_id = component_id
        self.container = dcc.Graph(
            id=component_id,
            className="workbench-container",
            figure=self.display_text("Waiting for Data..."),
            config={"scrollZoom": False, "doubleClick": "reset", "displayModeBar": False},
        )
        self.properties = [(self.component_id, "figure")]
        self.signals = [(self.component_id, "clickData")]
        return self.container

    def update_properties(self, df: pd.DataFrame, **kwargs) -> list:
        """Create a Proximity Circle Figure for the query target and its neighbors."""
        # Basic validation
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return [self.display_text("No Data")]

        # Get column names
        id_column = df.columns[0]
        target_column = df.columns[-1]

        # Ensure required columns exist
        required_columns = [id_column, "neighbor_id", "distance", target_column]
        if not all(col in df.columns for col in required_columns):
            return [self.display_text("Missing required data columns")]

        # Get query ID and its neighbors
        query_id = df.iloc[0][id_column]
        query_target = df[df["neighbor_id"] == query_id].iloc[0][target_column]
        neighbors = df[df["neighbor_id"] != query_id].sort_values("distance").head(6)

        # Create figure
        fig = go.Figure()

        # Center coordinates and layout parameters
        center = (0, 0)
        radius = 0.5

        # Calculate coordinates for surrounding circles
        circle_coords = []
        for i in range(min(6, len(neighbors))):
            angle = (np.pi / 3) * i
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            circle_coords.append((x, y))

        # Prepare data for visualization
        x_coords, y_coords = [], []
        lines_data = []
        node_data = []

        # Set up center node
        x_coords.append(center[0])
        y_coords.append(center[1])
        node_data.append(
            {
                "id": query_id,
                "color": 0,
                "size": 40,
                "hover_text": f"ID: {query_id}<br>Value: {query_target:.2f}",
                "text": f"{query_target:.2f}",
                "type": "query",
                "distance": 0,
                "target": query_target,
            }
        )

        # Set up neighbor nodes and connecting lines
        for i, (x, y) in enumerate(circle_coords):
            neighbor = neighbors.iloc[i]
            neighbor_id = neighbor["neighbor_id"]
            distance = neighbor["distance"]
            target_val = neighbor[target_column]
            delta = target_val - query_target

            # Add node data
            x_coords.append(x)
            y_coords.append(y)
            node_data.append(
                {
                    "id": neighbor_id,
                    "color": delta,
                    "size": 40,
                    "hover_text": f"ID: {neighbor_id}<br>Distance: {distance:.2f}<br>Value: {target_val:.2f}",
                    "text": f"{target_val:.2f}",
                    "type": "neighbor",
                    "distance": distance,
                    "target": target_val,
                }
            )

            # Add line data
            line_width = max(4 - distance, 0) * 5 + 5
            lines_data.append({"x": [center[0], x], "y": [center[1], y], "width": line_width})

        # Add connecting lines to figure
        for line in lines_data:
            fig.add_trace(
                go.Scatter(
                    x=line["x"],
                    y=line["y"],
                    mode="lines",
                    line=dict(color="rgba(150, 150, 150, 0.75)", width=line["width"]),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # Extract data for scatter plot
        colors = [node["color"] for node in node_data]
        sizes = [node["size"] for node in node_data]
        hover_texts = [node["hover_text"] for node in node_data]
        texts = [node["text"] for node in node_data]

        # Create custom data for interactivity
        custom_data = []
        for node in node_data:
            custom_data.append(
                {"type": node["type"], "id": node["id"], "distance": node["distance"], "target": node["target"]}
            )

        # Add nodes to figure
        scatter = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=dict(
                color=colors, colorscale=self.colorscale, size=sizes, line=dict(color="black", width=1), opacity=1.0
            ),
            text=texts,
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
        scatter.customdata = custom_data
        fig.add_trace(scatter)

        # Configure layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            width=200,
            showlegend=False,
            plot_bgcolor=self.theme_manager.background(),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            hovermode="closest",
            xaxis=dict(range=[-1, 1], visible=False),
            yaxis=dict(range=[-1, 1], visible=False, scaleanchor="x", scaleratio=1),
        )

        return [fig]

    def register_internal_callbacks(self):
        """Register internal callbacks for the plugin."""
        pass  # We'll implement this if needed


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Create sample data for testing
    test_data = pd.DataFrame(
        {
            "id": ["1", "1", "1", "1", "1", "1", "1"],
            "neighbor_id": ["1", "2", "3", "4", "5", "6", "7"],
            "distance": [0.0, 0.0, 1.299653, 1.377522, 1.442327, 2.636339, 5.636339],
            "activity": [4.2, 8.0, 3.499257, 4.368489, 4.34046, 8.0, 2.0],
        }
    )

    # Run the Unit Test on the Plugin
    PluginUnitTest(ProximityMiniGraph, input_data=test_data, theme="light").run()
