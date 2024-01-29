"""S3 Bucket Connector Tests"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.utils.config_manager import ConfigManager

# Grab out incoming data bucket for something to test with
cm = ConfigManager()
sageworks_bucket = cm.get_config("SAGEWORKS_BUCKET")
test_bucket = "s3://" + sageworks_bucket + "/incoming-data/"


def test_check():
    """Test the check() method"""
    bucket = S3Bucket(test_bucket)
    assert bucket.check() is True


def test_refresh():
    """Test the refresh() method"""
    bucket = S3Bucket(test_bucket)
    bucket.refresh()


def test_summary():
    """Test the summary() method"""
    bucket = S3Bucket(test_bucket)
    bucket.refresh()
    summary = bucket.summary()
    assert isinstance(summary, dict)
    assert any("incoming-data" in key for key in summary.keys())


if __name__ == "__main__":
    """Run the tests for the S3 Bucket Connector"""
    test_check()
    test_refresh()
    test_summary()
