"""Tests for confusion_matrix web component"""

# SageWorks Imports
from sageworks.web_components.confusion_matrix import ConfusionMatrix
from sageworks.api.model import Model


def test_confusion_matrix():
    """Test the ConfusionMatrix class"""
    # Instantiate model
    m = Model("wine-classification")
    inference_run = "training_holdout"

    # Instantiate the ConfusionMatrix class
    cm = ConfusionMatrix()

    # Generate the figure
    fig = cm.update_properties(m, inference_run)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()


if __name__ == "__main__":
    # Run the tests
    test_confusion_matrix()
