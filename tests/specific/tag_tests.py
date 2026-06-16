"""Tests for Workbench tag/metadata operations (add, get/list, delete)"""

import base64
import json

import pytest
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.utils.aws_utils import dict_to_aws_tags, aws_tags_to_dict, B64_MARKER


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
    """Values longer than one tag get chunked and must still round-trip (no AWS needed)"""
    meta = {
        "arns": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/abalone-regression," * 11,
        "big_list": [{"name": f"thing_{i}", "score": i} for i in range(60)],
    }
    assert aws_tags_to_dict(dict_to_aws_tags(meta)) == meta


def test_tag_safe_values_stored_plain():
    """Tag-safe values are stored as plain text (no b64: marker), unsafe values are marked"""
    tags = {t["key"]: t["value"] for t in dict_to_aws_tags({"safe": "us-west-2", "unsafe": "hello, world!"})}
    assert tags["safe"] == "us-west-2"  # plain, no marker
    assert tags["unsafe"].startswith(B64_MARKER)  # encoded + marked


def test_plain_non_base64_pass_through():
    """Plain/foreign tags that aren't valid base64 pass through untouched (no decode, no warning)"""
    aws_tags = [
        {"key": "status", "value": "ready"},  # plain word
        {"key": "owner", "value": "test_user"},  # underscore -> not valid base64
        {"key": "aws:cloudformation:stack-name", "value": "my-stack"},  # foreign AWS tag
    ]
    result = aws_tags_to_dict(aws_tags)
    assert result == {"status": "ready", "owner": "test_user", "aws:cloudformation:stack-name": "my-stack"}


def test_legacy_markerless_b64_decoded_transitional():
    """TRANSITIONAL: markerless-base64 tags are still decoded by the fallback.

    When _decode_legacy_b64 is removed, invert this: "TWFu" stays "TWFu" instead of decoding to "Man".
    """
    aws_tags = [{"key": "perm", "value": "TWFu"}]  # markerless base64 of "Man"
    assert aws_tags_to_dict(aws_tags) == {"perm": "Man"}


def test_marked_values_decode():
    """Values carrying the b64: marker are base64-decoded on read"""
    aws_tags = dict_to_aws_tags({"blob": {"a": 1, "b": "x!y"}})  # not tag-safe -> marked
    assert aws_tags[0]["value"].startswith(B64_MARKER)
    assert aws_tags_to_dict(aws_tags) == {"blob": {"a": 1, "b": "x!y"}}


def test_chunked_plain_value():
    """A long tag-safe value (>256) is chunked as PLAIN text (no marker) and round-trips"""
    long_plain = "a/b.c-d_e:" * 30  # 300 chars, all tag-safe, no comma
    tags = dict_to_aws_tags({"path": long_plain})
    assert any("_chunk_" in t["key"] for t in tags)  # it actually chunked
    assert not any(t["value"].startswith(B64_MARKER) for t in tags)  # no chunk is marked
    assert aws_tags_to_dict(tags)["path"] == long_plain


def test_chunked_marked_value():
    """A long unsafe value (>256) is b64:-marked, then chunked; marker stays on the first chunk only"""
    long_unsafe = "x!y," * 80  # 320 chars, '!' and ',' are not tag-safe
    tags = dict_to_aws_tags({"blob": long_unsafe})
    assert any("_chunk_" in t["key"] for t in tags)
    # Marker sits at the front of the reassembled string -> exactly one chunk carries it
    assert sum(t["value"].startswith(B64_MARKER) for t in tags) == 1
    assert aws_tags_to_dict(tags)["blob"] == long_unsafe


def test_legacy_chunked_b64_decoded_transitional():
    """TRANSITIONAL: old markerless-base64 values that were chunked still stitch + decode.

    Remove alongside the _decode_legacy_b64 fallback once migration is complete.
    """
    items = [{"i": i} for i in range(60)]
    # Old storage: a single key's value (here the list) base64-encoded, markerless, then chunked
    raw_b64 = base64.b64encode(json.dumps(items, separators=(",", ":")).encode()).decode()
    aws_tags = [{"key": f"items_chunk_{i + 1}", "value": raw_b64[i : i + 256]} for i in range(0, len(raw_b64), 256)]
    assert aws_tags_to_dict(aws_tags) == {"items": items}
