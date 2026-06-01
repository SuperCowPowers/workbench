# isort: skip_file
import logging
import sys
import types


class FakeWorkbenchCache:
    def __init__(self, *args, **kwargs):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    @staticmethod
    def flatten_key(name, *args, **kwargs):
        key_name = name.__name__ if callable(name) else name
        return f"{key_name}_{args}_{kwargs}"


fake_workbench_cache_module = types.ModuleType("workbench.utils.workbench_cache")
fake_workbench_cache_module.WorkbenchCache = FakeWorkbenchCache
sys.modules.setdefault("workbench.utils.workbench_cache", fake_workbench_cache_module)

from workbench.cached.cached_artifact_mixin import CachedArtifactMixin  # noqa: E402


class FakeMeta:
    def get_modified_timestamp(self, artifact):
        return "2026-06-01T00:00:00Z"


class FakeCache:
    def __init__(self, fail_on_set=False):
        self.fail_on_set = fail_on_set
        self.values = {}
        self.set_calls = 0

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.set_calls += 1
        if self.fail_on_set:
            raise RuntimeError("command not allowed when used memory > 'maxmemory'")
        self.values[key] = value


def install_fake_meta(monkeypatch):
    fake_meta_module = types.ModuleType("workbench.cached.cached_meta")
    fake_meta_module.CachedMeta = FakeMeta
    monkeypatch.setitem(sys.modules, "workbench.cached.cached_meta", fake_meta_module)


class DummyArtifact:
    name = "dummy"
    log = logging.getLogger("workbench")

    def __init__(self, value):
        self.value = value
        self.calls = 0

    @CachedArtifactMixin.cache_result
    def expensive_value(self):
        self.calls += 1
        return self.value


def test_oversized_result_is_returned_without_cache_write(monkeypatch):
    fake_cache = FakeCache()
    install_fake_meta(monkeypatch)
    monkeypatch.setattr(CachedArtifactMixin, "artifact_cache", fake_cache)
    monkeypatch.setattr(CachedArtifactMixin, "max_cached_result_bytes", 10)

    artifact = DummyArtifact("x" * 20)

    assert artifact.expensive_value() == "x" * 20
    assert artifact.calls == 1
    assert fake_cache.set_calls == 0


def test_cache_write_failure_returns_fresh_result(monkeypatch):
    fake_cache = FakeCache(fail_on_set=True)
    install_fake_meta(monkeypatch)
    monkeypatch.setattr(CachedArtifactMixin, "artifact_cache", fake_cache)
    monkeypatch.setattr(CachedArtifactMixin, "max_cached_result_bytes", 1024)

    artifact = DummyArtifact("fresh-result")

    assert artifact.expensive_value() == "fresh-result"
    assert artifact.calls == 1
    assert fake_cache.set_calls == 1


def test_cache_still_stores_small_results(monkeypatch):
    fake_cache = FakeCache()
    install_fake_meta(monkeypatch)
    monkeypatch.setattr(CachedArtifactMixin, "artifact_cache", fake_cache)
    monkeypatch.setattr(CachedArtifactMixin, "max_cached_result_bytes", 1024)

    artifact = DummyArtifact("small-result")

    assert artifact.expensive_value() == "small-result"
    assert artifact.expensive_value() == "small-result"
    assert artifact.calls == 1
    assert fake_cache.set_calls == 1
