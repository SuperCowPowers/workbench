"""DataCatalog Connector Tests"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.data_catalog import DataCatalog


def test_check():
    """Test the check() method"""
    catalog = DataCatalog()
    assert catalog.check() is True


def test_refresh():
    """Test the refresh() method"""
    catalog = DataCatalog()
    catalog.refresh()


def test_summary():
    """Test the summary() method"""
    catalog = DataCatalog()
    catalog.refresh()
    summary = catalog.summary()
    assert isinstance(summary, dict)
    assert "sageworks" in summary.keys()


def test_details():
    """Test the details() method"""
    catalog = DataCatalog()
    catalog.refresh()
    details = catalog.details("sageworks", "test_data")
    assert isinstance(details, dict)
    assert "Name" in details.keys()


if __name__ == "__main__":
    """Run the tests for the DataCatalog Connector"""
    test_check()
    test_refresh()
    test_summary()
    test_details()
