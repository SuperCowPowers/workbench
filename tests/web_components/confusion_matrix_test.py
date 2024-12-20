"""Tests for confusion_matrix web component"""

# Workbench Imports
from workbench.web_interface.components.plugins.confusion_matrix import ConfusionMatrix
from workbench.api.model import Model


def test_confusion_matrix():
    """Test the ConfusionMatrix class"""

    # Instantiate model
    model = Model("wine-classification")
    inference_run = "auto_inference"

    # Instantiate the ConfusionMatrix class
    cm = ConfusionMatrix()

    # Update the properties (the figure is the first element in the property list)
    all_properties = cm.update_properties(model, inference_run=inference_run)
    figure = all_properties[0]
    figure.show()


if __name__ == "__main__":
    # Run the tests
    test_confusion_matrix()
