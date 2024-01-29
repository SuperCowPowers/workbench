"""Model Registry Connector Tests"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.model_registry import ModelRegistry


def test_check():
    """Test the check() method"""
    model_registry = ModelRegistry()
    assert model_registry.check() is True


def test_refresh():
    """Test the refresh() method"""
    model_registry = ModelRegistry()
    model_registry.refresh()


def test_summary():
    """Test the summary() method"""
    model_registry = ModelRegistry()
    model_registry.refresh()
    summary = model_registry.summary()
    assert isinstance(summary, dict)
    assert "abalone-regression" in summary.keys()


if __name__ == "__main__":
    """Run the tests for the Model Registry Connector"""
    test_check()
    test_refresh()
    test_summary()
