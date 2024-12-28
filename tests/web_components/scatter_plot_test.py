"""Tests for graph_plot web component"""

# Workbench Imports
from workbench.api import FeatureSet
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot


def test_scatter_plot():
    """Test the ScatterPlot class"""

    # Instantiate a Graph
    fs = FeatureSet("abalone_features")

    # Instantiate the ScatterPlot class
    plot = ScatterPlot()

    # Update the properties (the figure is the first element in the property list)
    all_properties = plot.update_properties(fs.pull_dataframe())
    figure = all_properties[0]
    figure.show()


if __name__ == "__main__":
    # Run the tests
    test_scatter_plot()
