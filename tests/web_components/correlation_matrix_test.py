"""Tests for CorrelationMatrix web component"""

# Workbench Imports
from workbench.web_interface.components.correlation_matrix import CorrelationMatrix
from workbench.api.data_source import DataSource


def test_correlation_matrix():
    """Test the ConfusionMatrix class"""

    ds = DataSource("test_data")
    ds_details = ds.details()

    # Instantiate the CorrelationMatrix class
    corr_plot = CorrelationMatrix()

    # Generate the figure
    fig = corr_plot.update_properties(ds_details)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")
    return fig


if __name__ == "__main__":
    # Run the test and show the figure
    test_correlation_matrix().show()
