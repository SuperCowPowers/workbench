"""Tests for graph_plot web component"""

# SageWorks Imports
from sageworks.api import FeatureSet
from sageworks.web_components.plugins.scatter_plot import ScatterPlot


def test_scatter_plot():
    """Test the ScatterPlot class"""

    # Instantiate a Graph
    fs = FeatureSet("abalone_features")

    # Instantiate the ScatterPlot class
    plot = ScatterPlot()

    # Update the properties (the figure is the first element in the property list)
    all_properties = plot.update_properties(fs)
    figure = all_properties[0]
    figure.show()


if __name__ == "__main__":
    # Run the tests
    test_scatter_plot()
