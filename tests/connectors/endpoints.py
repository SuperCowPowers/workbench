"""Endpoints Connector Tests"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.endpoints import Endpoints


def test_check():
    """Test the check() method"""
    endpoints = Endpoints()
    assert endpoints.check() is True


def test_refresh():
    """Test the refresh() method"""
    endpoints = Endpoints()
    endpoints.refresh()


def test_summary():
    """Test the summary() method"""
    endpoints = Endpoints()
    endpoints.refresh()
    summary = endpoints.summary()
    assert isinstance(summary, dict)
    assert "abalone-regression-end" in summary.keys()


if __name__ == "__main__":
    """Run the tests for the Endpoints Connector"""
    test_check()
    test_refresh()
    test_summary()
