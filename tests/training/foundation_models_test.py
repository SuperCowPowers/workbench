"""Tests for the foundation checkpoint resolver: local cache -> account mirror -> public bucket."""

import hashlib

import pytest

from workbench.training import foundation_models as fm


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty local cache, and a record of every S3 download the resolver attempts."""
    monkeypatch.setattr(fm, "foundation_cache_dir", lambda: tmp_path)
    attempts = []
    monkeypatch.setattr(fm, "_download_s3", lambda bucket, key, dest, **kw: attempts.append((bucket, kw)) or False)
    return tmp_path, attempts


def test_a_cached_checkpoint_needs_no_download(cache):
    tmp_path, attempts = cache
    (tmp_path / fm.foundation_entry("monroe")["filename"]).write_bytes(b"cached")
    assert fm.resolve_foundation_checkpoint("monroe").read_bytes() == b"cached"
    assert attempts == []


def test_the_mirror_is_tried_before_the_public_bucket(cache, monkeypatch):
    _, attempts = cache
    monkeypatch.setattr(fm, "workbench_bucket", lambda: "my-account-bucket")
    with pytest.raises(RuntimeError, match=fm.PUBLIC_BUCKET):
        fm.resolve_foundation_checkpoint("tabicl")
    assert [(bucket, kw["unsigned"]) for bucket, kw in attempts] == [
        ("my-account-bucket", False),
        (fm.PUBLIC_BUCKET, True),
    ]
    assert all(kw["expected_md5"] == fm.foundation_entry("tabicl")["expected_md5"] for _, kw in attempts)


def test_without_a_mirror_the_public_bucket_is_read_anonymously(cache, monkeypatch):
    _, attempts = cache
    monkeypatch.setattr(fm, "workbench_bucket", lambda: None)
    with pytest.raises(RuntimeError):
        fm.resolve_foundation_checkpoint("chemeleon")
    assert [(bucket, kw["unsigned"]) for bucket, kw in attempts] == [(fm.PUBLIC_BUCKET, True)]


def test_a_download_that_fails_its_md5_never_enters_the_cache(tmp_path, monkeypatch):
    class FakeS3:
        def download_file(self, bucket, key, filename):
            open(filename, "wb").write(b"not the checkpoint")

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: FakeS3())
    dest = tmp_path / "weights.pt"
    assert not fm._download_s3("bucket", "key", dest, expected_md5="0" * 32)
    assert not dest.exists() and list(tmp_path.iterdir()) == []  # no half-downloads left behind

    good_md5 = hashlib.md5(b"not the checkpoint").hexdigest()
    assert fm._download_s3("bucket", "key", dest, expected_md5=good_md5)
    assert dest.read_bytes() == b"not the checkpoint"


def test_every_registered_checkpoint_is_published():
    """The public bucket holds each registered checkpoint at its key, at the registered size (anonymous read)"""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", region_name=fm.PUBLIC_BUCKET_REGION, config=Config(signature_version=UNSIGNED))
    for name in fm.known_foundation_models():
        entry = fm.foundation_entry(name)
        head = s3.head_object(Bucket=fm.PUBLIC_BUCKET, Key=entry["s3_key"])
        assert head["ContentLength"] == entry["expected_size_bytes"], name
