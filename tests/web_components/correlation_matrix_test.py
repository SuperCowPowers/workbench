"""Tests for CorrelationMatrix web component"""

# SageWorks Imports
from sageworks.web_components.correlation_matrix import CorrelationMatrix
from sageworks.api.data_source import DataSource


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

    # Show the figure
    fig.show()


if __name__ == "__main__":
    # Run the tests
    test_correlation_matrix()
