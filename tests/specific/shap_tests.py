"""Tests for the SHAP Feature Importance functionality"""

# Local Imports
from workbench.api import Model


def test_shap_values_reg():
    """Test retrieval of SHAP values for a regression model"""

    # Grab a test regression model
    model = Model("abalone-regression")

    # Verify SHAP methods don't crash (may return None for legacy models)
    importance = model.shap_importance()
    print(f"SHAP importance: {importance[:3] if importance else None}")

    values = model.shap_values()
    print(f"SHAP values shape: {values.shape if values is not None else None}")

    feature_vals = model.shap_feature_values()
    print(f"SHAP feature values shape: {feature_vals.shape if feature_vals is not None else None}")


def test_shap_values_class():
    """Test retrieval of SHAP values for a classification model"""

    # Grab a test classification model
    model = Model("wine-classification")

    # Verify SHAP methods don't crash (may return None for legacy models)
    importance = model.shap_importance()
    print(f"SHAP importance: {importance[:3] if importance else None}")

    values = model.shap_values()
    if isinstance(values, dict):
        print(f"SHAP values (multiclass): {list(values.keys())}")
    else:
        print(f"SHAP values: {values}")

    feature_vals = model.shap_feature_values()
    print(f"SHAP feature values shape: {feature_vals.shape if feature_vals is not None else None}")


if __name__ == "__main__":
    test_shap_values_reg()
    test_shap_values_class()
