"""FeatureStore Connector Tests"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.feature_store import FeatureStore


def test_check():
    """Test the check() method"""
    fs = FeatureStore()
    assert fs.check() is True


def test_refresh():
    """Test the refresh() method"""
    fs = FeatureStore()
    fs.refresh()


def test_summary():
    """Test the summary() method"""
    fs = FeatureStore()
    fs.refresh()
    summary = fs.summary()
    assert isinstance(summary, dict)
    assert "test_features" in summary.keys()


if __name__ == "__main__":
    """Run the tests for the Feature Store Connector"""
    test_check()
    test_refresh()
    test_summary()
