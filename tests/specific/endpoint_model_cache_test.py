"""Tests for EndpointCore's cached ModelCore lookup."""

from workbench.core.artifacts import endpoint_core
from workbench.core.artifacts.endpoint_core import EndpointCore


def _endpoint_for_cache_test(model_name: str) -> EndpointCore:
    endpoint = EndpointCore.__new__(EndpointCore)
    endpoint.name = "test-endpoint"
    endpoint.model_name = model_name
    endpoint._model_core = None
    endpoint._model_core_name = None
    endpoint.get_input = lambda: model_name
    return endpoint


def test_model_core_is_cached(monkeypatch):
    created = []

    class FakeModelCore:
        def __init__(self, model_name):
            self.model_name = model_name
            created.append(model_name)

    monkeypatch.setattr(endpoint_core, "ModelCore", FakeModelCore)
    endpoint = _endpoint_for_cache_test("model-a")

    first = endpoint._model()
    second = endpoint._model()

    assert first is second
    assert first.model_name == "model-a"
    assert created == ["model-a"]


def test_model_cache_is_invalidated_when_model_changes(monkeypatch):
    created = []

    class FakeModelCore:
        def __init__(self, model_name):
            self.model_name = model_name
            created.append(model_name)

    monkeypatch.setattr(endpoint_core, "ModelCore", FakeModelCore)
    endpoint = _endpoint_for_cache_test("model-a")

    first = endpoint._model()
    endpoint._set_model_name("model-b")
    second = endpoint._model()

    assert first is not second
    assert second.model_name == "model-b"
    assert created == ["model-a", "model-b"]
