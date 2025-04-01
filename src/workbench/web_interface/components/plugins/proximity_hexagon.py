"""A proximity hexagon plugin component for comparing a query target to its neighbors"""

from dash import dcc, callback, Output, Input, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager


class ProximityHexagon(PluginInterface):
    """Proximity Hexagon Component for visualizing a target and its nearest neighbors"""

    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def __init__(self):
        """Initialize the ProximityHexagon plugin class"""
        self.component_id = None
        self.current_highlight = None
        self.theme_manager = ThemeManager()
        self.colorscale = self.theme_manager.colorscale("diverging")
        super().__init__()

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Proximity Hexagon Component without any data."""
        self.component_id = component_id
        self.container = dcc.Graph(
            id=component_id,
            className="workbench-container",
            figure=self.display_text("Waiting for Data..."),
            config={"scrollZoom": False, "doubleClick": "reset"},
        )
        self.properties = [(self.component_id, "figure")]
        self.signals = [(self.component_id, "clickData")]
        return self.container

    def update_properties(self, df: pd.DataFrame, **kwargs) -> list:
        """Create a Proximity Hexagon Figure for the query target and its neighbors."""
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

        # Helper function to create a hexagon
        def create_hexagon(x_center, y_center, size=0.25, hover_text=None):
            hex_points = []
            for i in range(6):
                angle = (np.pi / 3) * i + np.pi / 6
                x = x_center + size * np.cos(angle)
                y = y_center + size * np.sin(angle)
                hex_points.append((x, y))
            hex_points.append(hex_points[0])  # Close the shape
            x_vals, y_vals = zip(*hex_points)
            return go.Scatter(
                x=x_vals, y=y_vals,
                fill="toself",
                line=dict(color="rgba(100, 100, 100, 1)", width=1),
                mode="lines",
                text=hover_text,
                hoverinfo="text" if hover_text else "none",
                showlegend=False
            )

        # Add center hexagon
        center = (0, 0)
        center_hex = create_hexagon(
            center[0],
            center[1],
            hover_text=f"ID: {query_id}"
        )
        center_hex.customdata = [{"type": "query", "id": query_id, "target": query_target}]
        fig.add_trace(center_hex)

        # Add center label (with no hover)
        fig.add_trace(go.Scatter(
            x=[center[0]], y=[center[1]],
            mode="text",
            text=[f"{query_target:.2f}"],
            textposition="middle center",
            hoverinfo="none",
            showlegend=False
        ))

        # Calculate coordinates for surrounding hexagons
        radius = 0.43
        hex_coords = []
        for i in range(6):
            angle = (np.pi / 3) * i
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            hex_coords.append((x, y))

        # Add neighbor hexagons
        for i, (x, y) in enumerate(hex_coords):
            if i < len(neighbors):
                neighbor = neighbors.iloc[i]
                neighbor_id = neighbor["neighbor_id"]
                distance = neighbor["distance"]
                target_val = neighbor[target_column]
                delta = target_val - query_target

                # Create neighbor hexagon with hover text
                hover_text = f"ID: {neighbor_id}<br>Distance: {distance:.2f}"
                neighbor_hex = create_hexagon(x, y, hover_text=hover_text)
                neighbor_hex.customdata = [{"type": "neighbor", "id": neighbor_id, "distance": distance, "target": target_val}]
                fig.add_trace(neighbor_hex)

                # Add Value label (with no hover)
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="text",
                    text=[f"{target_val:.2f}"],
                    textposition="middle center",
                    hoverinfo="none",
                    showlegend=False
                ))

        # Configure layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300, width=300,
            showlegend=False,
            plot_bgcolor=self.theme_manager.background()
        )

        # Configure fixed aspect ratio and view range
        fig.update_layout(
            xaxis=dict(
                range=[-1, 1],
                visible=False
            ),
            yaxis=dict(
                range=[-1, 1],
                visible=False,
                scaleanchor="x",
                scaleratio=1
            )
        )

        return [fig]

    def register_internal_callbacks(self):
        """Register internal callbacks for the plugin."""
        # TBD: Implement highlight hexagon callback


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Create sample data for testing
    test_data = pd.DataFrame({
        "id": ["1", "1", "1", "1", "1", "1", "1"],
        "neighbor_id": ["1", "2", "3", "4", "5", "6", "7"],
        "distance": [0.0, 0.0, 1.299653, 1.377522, 1.442327, 1.636339, 1.636339],
        "activity": [-4.699482, -8.0, -3.499257, -4.368489, -4.34046, -8.0, -2.0]
    })

    # Run the Unit Test on the Plugin
    PluginUnitTest(ProximityHexagon, input_data=test_data, theme="light").run()