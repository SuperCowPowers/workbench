"""Tests for the SHAP Feature Importance functionality"""

# Local Imports
from sageworks.api import Endpoint


def test_generate_shap_values_reg():
    """Test the generation of SHAP values for a regression model"""

    # Grab a test regression endpoint
    end = Endpoint("abalone-regression-end")

    # Running auto_inference with capture=True will invoke the SHAP values generation
    print("Running SHAP values generation for regression")
    end.auto_inference(capture=True)


def test_generate_shap_values_class():
    """Test the generation of SHAP values for a classification model"""

    # Grab a test classification endpoint
    end = Endpoint("wine-classification-end")

    # Running auto_inference with capture=True will invoke the SHAP values generation
    print("Running SHAP values generation for classification")
    end.auto_inference(capture=True)


if __name__ == "__main__":
    test_generate_shap_values_reg()
    test_generate_shap_values_class()
