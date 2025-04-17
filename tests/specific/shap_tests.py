"""Tests for the SHAP Feature Importance functionality"""

# Local Imports
from workbench.api import Model


def test_generate_shap_values_reg():
    """Test the generation of SHAP values for a regression model"""

    # Grab a test regression model
    model = Model("abalone-regression")
    model.compute_shap_values()


def test_generate_shap_values_class():
    """Test the generation of SHAP values for a classification model"""

    # Grab a test classification model
    model = Model("wine-classification")
    model.compute_shap_values()


if __name__ == "__main__":
    test_generate_shap_values_reg()
    test_generate_shap_values_class()
