"""Tests for graph_plot web component"""

import time

# Workbench Imports
from workbench.api.graph_store import GraphStore
from workbench.web_interface.components.plugins.graph_plot import GraphPlot
from workbench.utils.theme_manager import ThemeManager


def test_graph_plot():
    """Test the GraphPlot class"""

    # Set the theme
    ThemeManager().set_theme("light")

    # Instantiate a Graph
    graph = GraphStore().get("test/karate_club")

    # Instantiate the GraphPlot class
    graph_plot = GraphPlot()

    # Generate the figure
    properties = graph_plot.update_properties(graph, label="club")

    # The first property should be the figure
    figure = properties[0]
    figure.show()

    # Sleep for a second (for plot to show)
    time.sleep(1)


if __name__ == "__main__":
    # Run the tests
    test_graph_plot()
