import workbench.core.artifacts.endpoint_core as endpoint_module

EndpointCore = endpoint_module.EndpointCore


def test_endpoint_delete_passes_model_name_to_managed_delete(monkeypatch):
    captured = {}

    def fake_managed_delete(cls, endpoint_name, model_name=None):
        captured["endpoint_name"] = endpoint_name
        captured["model_name"] = model_name

    monkeypatch.setattr(EndpointCore, "managed_delete", classmethod(fake_managed_delete))

    endpoint = EndpointCore.__new__(EndpointCore)
    endpoint.name = "demo-endpoint"
    endpoint.model_name = "demo-model"
    endpoint.exists = lambda: True

    endpoint.delete()

    assert captured == {"endpoint_name": "demo-endpoint", "model_name": "demo-model"}


def test_managed_delete_deregisters_model_from_endpoint_tags(monkeypatch):
    calls = []

    class FakeEndpoint:
        endpoint_config_name = "demo-config"
        async_inference_config = None

        def delete(self):
            calls.append(("endpoint.delete",))

    class FakeConfig:
        def delete(self):
            calls.append(("config.delete",))

    class FakeCache:
        def delete_recursive(self, endpoint_name):
            calls.append(("cache.delete_recursive", endpoint_name))

    class FakeModelCore:
        def __init__(self, model_name):
            self.model_name = model_name

        def remove_endpoint(self, endpoint_name):
            calls.append(("model.remove_endpoint", self.model_name, endpoint_name))

    monkeypatch.setattr(endpoint_module.SagemakerEndpoint, "get", staticmethod(lambda *args, **kwargs: FakeEndpoint()))
    monkeypatch.setattr(
        endpoint_module.SagemakerEndpointConfig, "get", staticmethod(lambda *args, **kwargs: FakeConfig())
    )
    monkeypatch.setattr(endpoint_module.MonitoringSchedule, "get_all", staticmethod(lambda *args, **kwargs: []))
    monkeypatch.setattr(endpoint_module, "ModelCore", FakeModelCore)
    monkeypatch.setattr(EndpointCore, "df_cache", FakeCache())
    monkeypatch.setattr(
        EndpointCore, "delete_endpoint_models", classmethod(lambda cls, name: calls.append(("models.delete", name)))
    )
    monkeypatch.setattr(
        EndpointCore,
        "_model_name_from_endpoint_tags",
        classmethod(lambda cls, endpoint_name: "demo-model"),
    )
    monkeypatch.setattr(endpoint_module.wr.s3, "list_objects", lambda *args, **kwargs: [])
    monkeypatch.setattr(endpoint_module.time, "sleep", lambda seconds: None)

    EndpointCore.managed_delete("demo-endpoint")

    assert ("model.remove_endpoint", "demo-model", "demo-endpoint") in calls
    assert ("models.delete", "demo-endpoint") in calls
    assert ("endpoint.delete",) in calls
