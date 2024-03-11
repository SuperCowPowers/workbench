"""Tests for inference run selector web component"""

# SageWorks Imports
from sageworks.web_components.inference_run_selector import InferenceRunSelector
from sageworks.api.model import Model


def test_inference_run_selector():
    """Test the ConfusionMatrix class"""
    # Instantiate model
    m = Model("wine-classification")
    inference_run = "training_holdout"

    # Instantiate the ConfusionMatrix class
    irs = InferenceRunSelector()

    # Create the component
    dropdown = irs.create_component("dropdown")

    # TBD
    print(m)
    print(inference_run)
    print(dropdown)


if __name__ == "__main__":
    # Run the tests
    test_inference_run_selector()
