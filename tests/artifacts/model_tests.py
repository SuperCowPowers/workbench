"""Tests for the Model functionality"""

# Workbench Imports
from workbench.core.artifacts.model_core import ModelCore


def test():
    """Simple test of the Model functionality"""

    # Grab a Model object and pull some information from it
    my_model = ModelCore("abalone-regression")

    # Call the various methods

    # Let's do a check/validation of the Model
    print(f"Model Check: {my_model.exists()}")

    # Get the ARN of the Model Group
    print(f"Model Group ARN: {my_model.group_arn()}")
    print(f"Model Package ARN: {my_model.arn()}")

    # Get the tags associated with this Model
    print(f"Tags: {my_model.get_tags()}")

    # Get creation time
    print(f"Created: {my_model.created()}")

    # Delete the Model
    # ModelCore.delete("abalone-regression")


if __name__ == "__main__":
    test()
