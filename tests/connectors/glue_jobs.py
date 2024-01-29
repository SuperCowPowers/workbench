"""Glue Jobs Connector Tests"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.glue_jobs import GlueJobs


def test_check():
    """Test the check() method"""
    jobs = GlueJobs()
    assert jobs.check() is True


def test_refresh():
    """Test the refresh() method"""
    jobs = GlueJobs()
    jobs.refresh()


def test_summary():
    """Test the summary() method"""
    jobs = GlueJobs()
    jobs.refresh()
    summary = jobs.summary()
    assert isinstance(summary, dict)
    assert "Glue_Test" in summary.keys()


def test_details():
    """Test the details() method"""
    jobs = GlueJobs()
    jobs.refresh()
    details = jobs.details("Glue_Test")
    assert isinstance(details, dict)
    assert "DefaultArguments" in details.keys()


if __name__ == "__main__":
    """Run the tests for the Glue Jobs Connector"""
    test_check()
    test_refresh()
    test_summary()
    test_details()
