"""Tests for graph_plot web component"""

# Workbench Imports
from workbench.utils.graph_utils import load_graph
from workbench.web_interface.components.plugins.graph_plot import GraphPlot


def test_graph_plot():
    """Test the GraphPlot class"""

    # Instantiate a Graph
    graph = load_graph("karate_club")

    # Instantiate the GraphPlot class
    graph_plot = GraphPlot()

    # Generate the figure
    properties = graph_plot.update_properties(graph, label="club")

    # The first property should be the figure
    figure = properties[0]
    figure.show()


if __name__ == "__main__":
    # Run the tests
    test_graph_plot()
