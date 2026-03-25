"""Tests for SageMaker V3 Tag operations (add, get/list, delete)"""

import pytest
from sagemaker.core.resources import Tag
from sagemaker.core.common_utils import list_tags
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.utils.aws_utils import dict_to_aws_tags, aws_tags_to_dict


@pytest.fixture(scope="module")
def fs():
    """Get a FeatureSetCore object for testing"""
    return FeatureSetCore("abalone_features")


def test_tag_add(fs):
    """Test adding tags to a SageMaker resource via V3 Tag.add_tags()"""
    arn = fs.arn()
    assert arn is not None

    # Add tags using V3 lowercase dict format
    tags = [{"key": "test_add_tag", "value": "hello_world"}]
    Tag.add_tags(resource_arn=arn, tags=tags, session=fs.boto3_session)

    # Verify the tag was added by reading raw AWS tags
    raw_tags = list_tags(fs.sm_session, arn)
    tag_dict = {t["Key"]: t["Value"] for t in raw_tags}
    assert "test_add_tag" in tag_dict
    assert tag_dict["test_add_tag"] == "hello_world"


def test_tag_list(fs):
    """Test listing/getting tags from a SageMaker resource via list_tags"""
    arn = fs.arn()
    raw_tags = list_tags(fs.sm_session, arn)

    # Should be a list of dicts with Key/Value
    assert isinstance(raw_tags, list)
    assert len(raw_tags) > 0

    tag_dict = {t["Key"]: t["Value"] for t in raw_tags}
    # Workbench always has workbench_tags
    assert "workbench_tags" in tag_dict or "workbench_status" in tag_dict


def test_tag_delete(fs):
    """Test deleting tags from a SageMaker resource via V3 Tag.delete_tags()"""
    arn = fs.arn()

    # First add a tag we can delete
    tags = [{"key": "test_delete_tag", "value": "to_be_deleted"}]
    Tag.add_tags(resource_arn=arn, tags=tags, session=fs.boto3_session)

    # Verify it exists
    raw_tags = list_tags(fs.sm_session, arn)
    tag_dict = {t["Key"]: t["Value"] for t in raw_tags}
    assert "test_delete_tag" in tag_dict

    # Now delete it
    Tag.delete_tags(resource_arn=arn, tag_keys=["test_delete_tag"], session=fs.boto3_session)

    # Verify it's gone
    raw_tags = list_tags(fs.sm_session, arn)
    tag_dict = {t["Key"]: t["Value"] for t in raw_tags}
    assert "test_delete_tag" not in tag_dict


def test_tag_cleanup(fs):
    """Clean up: remove the test_add_tag we created"""
    arn = fs.arn()
    Tag.delete_tags(resource_arn=arn, tag_keys=["test_add_tag"], session=fs.boto3_session)


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
