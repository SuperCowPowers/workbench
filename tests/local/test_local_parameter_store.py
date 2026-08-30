"""Tests for the filesystem-backed ParameterStore in workbench.local.

Covers the behavior the AWS store promises and this one has to reproduce: names
are canonicalized the same way, prefixes match whole path segments, and values
round-trip through JSON. Nothing here touches AWS.
"""

import json
import os
from datetime import datetime, timezone

import pytest

from workbench.local import parameter_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A ParameterStore rooted in a throwaway directory."""
    monkeypatch.setattr(parameter_store, "local_root", lambda: str(tmp_path))
    return parameter_store.ParameterStore()


#
# Names
#


@pytest.mark.parametrize(
    "given, expected",
    [
        ("workbench/test", "/workbench/test"),
        ("/workbench/test", "/workbench/test"),
        ("//workbench//test/", "/workbench/test"),
        ("/workbench/test/", "/workbench/test"),
    ],
)
def test_normalize(given, expected):
    """Names are absolute, single-slashed, and carry no trailing slash"""
    assert parameter_store.ParameterStore._normalize(given) == expected


def test_leading_slash_is_optional(store):
    """The same parameter is reachable with or without the leading slash"""
    store.upsert("workbench/test", "value")
    assert store.get("/workbench/test") == "value"


@pytest.mark.parametrize("name", ["../escape", "/workbench/../../escape", "/", ""])
def test_names_cannot_escape_the_root(store, name):
    """A parameter name is not a path into the rest of the filesystem"""
    with pytest.raises(ValueError):
        store.upsert(name, "value")


#
# Round-trips
#


@pytest.mark.parametrize(
    "value",
    ["a string", 42, 4.2, ["a", "b"], {"key": "value", "number": 4.2, "list": [1, 2, 3]}],
)
def test_value_round_trips(store, value):
    """Values come back as the type they went in as"""
    store.upsert("/workbench/test", value)
    assert store.get("/workbench/test") == value


def test_upsert_overwrites(store):
    """A second upsert replaces the value rather than appending"""
    store.upsert("/workbench/test", "first")
    store.upsert("/workbench/test", "second")
    assert store.get("/workbench/test") == "second"


def test_backing_file_layout(store, tmp_path):
    """The parameter path becomes the directory path, so names map somewhere predictable"""
    store.upsert("/workbench/bosco/sessions/briford/logd", "report")
    expected = tmp_path / parameter_store.SUBDIR / "workbench/bosco/sessions/briford/logd.json"
    assert expected.is_file()


def test_missing_parameter_is_none(store):
    """A parameter that was never written reads back as None"""
    assert store.get("/workbench/nope") is None
    assert store.get("/workbench/nope", warn=False) is None


def test_non_json_content_returns_raw(store, tmp_path):
    """Content that isn't JSON is handed back as-is rather than raising"""
    path = tmp_path / parameter_store.SUBDIR / "workbench"
    path.mkdir(parents=True)
    (path / "raw.json").write_text("not json at all")
    assert store.get("/workbench/raw") == "not json at all"


#
# Listing
#


def test_list_is_sorted_names(store):
    """Bare listing returns canonical names in sorted order"""
    for name in ["/workbench/b", "/workbench/a", "/other/c"]:
        store.upsert(name, "v")
    assert store.list() == ["/other/c", "/workbench/a", "/workbench/b"]


def test_list_empty_store(store):
    """A store with nothing in it lists nothing (and does not create the root)"""
    assert store.list() == []


def test_list_details(store):
    """Details mode carries the modification time alongside the name"""
    store.upsert("/workbench/test", "v")
    (entry,) = store.list(details=True)
    assert entry["name"] == "/workbench/test"
    assert isinstance(entry["modified"], datetime)
    assert entry["modified"].tzinfo is not None


def test_list_prefix_matches_whole_segments(store):
    """A prefix matches path segments, not arbitrary string prefixes"""
    store.upsert("/workbench/models/one", "v")
    store.upsert("/workbench/models_other/two", "v")
    assert store.list(prefix="/workbench/models") == ["/workbench/models/one"]


def test_list_prefix_leading_slash_optional(store):
    """The prefix is canonicalized the same way a name is"""
    store.upsert("/workbench/models/one", "v")
    assert store.list(prefix="workbench/models") == ["/workbench/models/one"]


#
# Timestamps and deletion
#


def test_last_modified(store):
    """A written parameter reports when it was written"""
    store.upsert("/workbench/test", "v")
    when = store.last_modified("/workbench/test")
    assert isinstance(when, datetime)
    assert (datetime.now(timezone.utc) - when).total_seconds() < 60


def test_last_modified_missing_is_none(store):
    """A parameter that doesn't exist has no timestamp"""
    assert store.last_modified("/workbench/nope") is None


def test_delete(store):
    """Delete removes the parameter"""
    store.upsert("/workbench/test", "v")
    store.delete("/workbench/test")
    assert store.get("/workbench/test", warn=False) is None


def test_delete_missing_does_not_raise(store):
    """Deleting something absent is logged, not raised -- same as the AWS store"""
    store.delete("/workbench/nope")


def test_delete_recursive(store):
    """Recursive delete takes everything under the path"""
    store.upsert("/workbench/models/one", "v")
    store.upsert("/workbench/models/two", "v")
    store.upsert("/workbench/other", "v")
    store.delete_recursive("/workbench/models")
    assert store.list() == ["/workbench/other"]


def test_delete_recursive_spares_the_path_itself(store):
    """A parameter named exactly the prefix is a sibling of that path, not a child"""
    store.upsert("/workbench/models", "the path itself")
    store.upsert("/workbench/models/one", "a child")
    store.delete_recursive("/workbench/models")
    assert store.get("/workbench/models") == "the path itself"


def test_delete_recursive_prunes_empty_directory(store, tmp_path):
    """The directory goes away once nothing is left in it"""
    store.upsert("/workbench/models/one", "v")
    store.delete_recursive("/workbench/models")
    assert not os.path.isdir(tmp_path / parameter_store.SUBDIR / "workbench/models")


#
# Encoding
#


def test_json_encoder_handles_numpy(store, tmp_path):
    """Values go through the same encoder the AWS store uses"""
    np = pytest.importorskip("numpy")
    store.upsert("/workbench/test", {"count": np.int64(7), "score": np.float64(0.5)})
    assert store.get("/workbench/test") == {"count": 7, "score": 0.5}
    written = (tmp_path / parameter_store.SUBDIR / "workbench/test.json").read_text()
    assert json.loads(written) == {"count": 7, "score": 0.5}
