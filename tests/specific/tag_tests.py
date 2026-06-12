"""Tests for Workbench tag/metadata operations (add, get/list, delete)"""

import pytest
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.utils.aws_utils import dict_to_aws_tags, aws_tags_to_dict


@pytest.fixture(scope="module")
def fs():
    """Get a FeatureSetCore object for testing"""
    return FeatureSetCore("abalone_features")


def test_tag_add(fs):
    """Test adding metadata via upsert_workbench_meta()"""
    fs.upsert_workbench_meta({"test_add_tag": "hello_world"})

    # Verify the tag was added (workbench_meta reads AWS tags with throttle backoff)
    meta = fs.workbench_meta()
    assert meta.get("test_add_tag") == "hello_world"


def test_tag_list(fs):
    """Test listing metadata via workbench_meta()"""
    meta = fs.workbench_meta()
    assert isinstance(meta, dict)
    assert len(meta) > 0

    # Workbench always has workbench_tags
    assert "workbench_tags" in meta or "workbench_status" in meta


def test_tag_delete(fs):
    """Test deleting metadata via delete_metadata()"""

    # First add a tag we can delete
    fs.upsert_workbench_meta({"test_delete_tag": "to_be_deleted"})
    assert "test_delete_tag" in fs.workbench_meta()

    # Now delete it
    fs.delete_metadata("test_delete_tag")
    assert "test_delete_tag" not in fs.workbench_meta()


def test_tag_cleanup(fs):
    """Clean up: remove the test_add_tag we created"""
    fs.delete_metadata("test_add_tag")


def test_dict_to_aws_tags():
    """Test the dict_to_aws_tags conversion (unit test, no AWS needed)"""
    meta = {"status": "ready", "owner": "test_user"}
    tags = dict_to_aws_tags(meta)

    # Should produce lowercase key/value dicts
    assert len(tags) == 2
    for tag in tags:
        assert "key" in tag
        assert "value" in tag
        assert "Key" not in tag  # No uppercase
        assert "Value" not in tag


def test_aws_tags_to_dict():
    """Test the aws_tags_to_dict conversion (unit test, no AWS needed)"""
    # V3 format (lowercase)
    aws_tags = [{"key": "status", "value": "ready"}, {"key": "owner", "value": "test_user"}]
    result = aws_tags_to_dict(aws_tags)
    assert result["status"] == "ready"
    assert result["owner"] == "test_user"


def test_tag_roundtrip_value_types():
    """Mixed value types survive the encode -> tags -> decode round-trip (no AWS needed)"""
    meta = {
        "status": "ready",
        "count": 42,
        "ratios": [0.1, 0.2, 0.3],
        "nested": {"a": [1, 2, 3], "b": "hello world!"},
        "empty": "",
    }
    assert aws_tags_to_dict(dict_to_aws_tags(meta)) == meta


def test_tag_roundtrip_chunked():
    """Values longer than one tag get chunked and must still round-trip (no AWS needed).

    Regression: a long tag-safe value (joined ARNs) previously stitched back to a base64
    string of length 4n+1 and raised binascii.Error on read.
    """
    meta = {
        "arns": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/abalone-regression," * 11,
        "big_list": [{"name": f"thing_{i}", "score": i} for i in range(60)],
    }
    assert aws_tags_to_dict(dict_to_aws_tags(meta)) == meta


def test_decode_legacy_unencoded_tags():
    """Legacy tags written before base64-everything (stored raw) still decode (no AWS needed)"""
    aws_tags = [
        {"key": "status", "value": "ready"},  # plain string
        {"key": "owner", "value": "test_user"},  # underscores, not base64
    ]
    result = aws_tags_to_dict(aws_tags)
    assert result == {"status": "ready", "owner": "test_user"}
