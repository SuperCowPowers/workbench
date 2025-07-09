"""Tests for Endpoint 'chaining''"""

# Workbench Imports
from workbench.api import Endpoint

end_reg = Endpoint("test-regression")
end_class = Endpoint("test-classification")


def test_reg_to_reg():
    """Test chaining two regression endpoints"""
    pred_df = end_reg.inference(end_reg.auto_inference())
    print(pred_df.columns)
    print(pred_df)


def test_reg_to_class():
    """Test chaining a regression endpoint to a classification endpoint"""
    pred_df = end_class.inference(end_reg.auto_inference())
    print(pred_df.columns)
    print(pred_df)


def test_class_to_class():
    """Test chaining two classification endpoints"""
    pred_df = end_class.inference(end_class.auto_inference())
    print(pred_df.columns)
    print(pred_df)


def test_class_to_reg():
    """Test chaining a classification endpoint to a regression endpoint"""
    pred_df = end_reg.inference(end_class.auto_inference())
    print(pred_df.columns)
    print(pred_df)


if __name__ == "__main__":

    # Set Pandas Display Options
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Run the tests
    test_reg_to_reg()
    test_reg_to_class()
    test_class_to_class()
    test_class_to_reg()

    print("All tests passed!")
